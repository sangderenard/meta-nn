/*
 * nodus_loss_store.c -- Implementation of the loss data repository.
 *
 * Cross-process shared-memory store with mutex-protected access.
 * The global singleton (get_global) is backed by OS named shared memory
 * so every process that loads this DLL sees the SAME physical data.
 *
 * Synchronisation:
 *   Windows -- named mutex (CreateMutexA) for the shared-memory global;
 *              CRITICAL_SECTION for heap-allocated stores (single-process).
 *   POSIX   -- pthread_mutex with PTHREAD_PROCESS_SHARED in shared memory;
 *              plain pthread_mutex for heap-allocated stores.
 */

#ifndef NODUS_BUILDING_DLL
#  define NODUS_BUILDING_DLL
#endif
#include "nodus_loss_store.h"
#include "nodus_weight_image.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Forward declaration -- needed by store_lock / store_unlock. */
static NodusLossStore *g_global;

/* --- Platform mutex type (must precede struct definition) --- */
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   typedef CRITICAL_SECTION nodus_mutex_t;
   static HANDLE g_shm_mutex = NULL;  /* named mutex for the global store */

   static void mutex_init(nodus_mutex_t *m)    { InitializeCriticalSection(m); }
   static void mutex_destroy(nodus_mutex_t *m) { DeleteCriticalSection(m); }
#else
#  include <pthread.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
   typedef pthread_mutex_t nodus_mutex_t;

   static void mutex_init(nodus_mutex_t *m) {
       pthread_mutexattr_t attr;
       pthread_mutexattr_init(&attr);
       pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
       pthread_mutex_init(m, &attr);
       pthread_mutexattr_destroy(&attr);
   }
   static void mutex_init_shared(nodus_mutex_t *m) {
       pthread_mutexattr_t attr;
       pthread_mutexattr_init(&attr);
       pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
       pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
       pthread_mutex_init(m, &attr);
       pthread_mutexattr_destroy(&attr);
   }
   static void mutex_destroy(nodus_mutex_t *m) { pthread_mutex_destroy(m); }
#endif

/* ------------------------------------------------------------------ */
/*  Internal channel structure                                        */
/* ------------------------------------------------------------------ */

typedef struct NodusChannel {
    char             name[NODUS_MAX_CHANNEL_NAME];
    int32_t          capacity;      /* max_records_per_channel */
    int32_t          length;        /* current count */
    int32_t          step_cursor;   /* next step to assign */
    int32_t          head;          /* index of oldest record in circular buffer */
    NodusLossRecord  records[NODUS_MAX_RECORDS]; /* inline record storage */
} NodusChannel;

struct NodusLossStore {
    nodus_mutex_t   lock;
    int             max_channels;
    int             max_records;
    int             channel_count;
    int32_t         initialized;    /* magic sentinel: 0x4E4F4455 when valid */
    NodusChannel    channels[NODUS_MAX_CHANNELS]; /* inline channel storage */
};

/* --- Platform lock/unlock routing (needs full struct definition) --- */
#ifdef _WIN32
   /* Route to named mutex for the global store, CRITICAL_SECTION otherwise. */
   static void store_lock(NodusLossStore *store) {
       if (store == g_global && g_shm_mutex)
           WaitForSingleObject(g_shm_mutex, INFINITE);
       else
           EnterCriticalSection(&store->lock);
   }
   static void store_unlock(NodusLossStore *store) {
       if (store == g_global && g_shm_mutex)
           ReleaseMutex(g_shm_mutex);
       else
           LeaveCriticalSection(&store->lock);
   }
#else
   /* On POSIX the mutex lives in shared memory -- always use it directly. */
   static void store_lock(NodusLossStore *store)   { pthread_mutex_lock(&store->lock); }
   static void store_unlock(NodusLossStore *store) { pthread_mutex_unlock(&store->lock); }
#endif

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Find channel index by name.  Returns -1 if not found. */
static int find_channel(const NodusLossStore *store, const char *key) {
    for (int i = 0; i < store->channel_count; i++) {
        if (strncmp(store->channels[i].name, key, NODUS_MAX_CHANNEL_NAME) == 0)
            return i;
    }
    return -1;
}

/* Compute the actual buffer index from a logical position. */
static int buf_index(const NodusChannel *ch, int logical) {
    return (ch->head + logical) % ch->capacity;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/* Initialise a NodusLossStore struct in-place (used by both create and
   the shared-memory global).  Caller must have zeroed the memory.
   When cross_process is true the mutex is set up for inter-process use
   (POSIX: PTHREAD_PROCESS_SHARED; Windows: handled externally via named mutex). */
static void store_init(NodusLossStore *store,
                       int max_channels, int max_records_per_channel,
                       int cross_process)
{
#ifdef _WIN32
    /* Windows: CRITICAL_SECTION is initialised but only used for heap stores.
       For the shared-memory global the named mutex is used instead. */
    (void)cross_process;
    mutex_init(&store->lock);
#else
    if (cross_process)
        mutex_init_shared(&store->lock);
    else
        mutex_init(&store->lock);
#endif
    store->max_channels  = max_channels;
    store->max_records   = max_records_per_channel;
    store->channel_count = 0;
    for (int i = 0; i < max_channels; i++) {
        store->channels[i].capacity    = max_records_per_channel;
        store->channels[i].length      = 0;
        store->channels[i].step_cursor = 0;
        store->channels[i].head        = 0;
        store->channels[i].name[0]     = '\0';
    }
    store->initialized = 0x4E4F4455; /* 'NODU' */
}

NODUS_API NodusLossStore* nodus_loss_store_create(
        int max_channels, int max_records_per_channel)
{
    max_channels            = clamp_i(max_channels, 1, NODUS_MAX_CHANNELS);
    max_records_per_channel = clamp_i(max_records_per_channel, 1, NODUS_MAX_RECORDS);

    NodusLossStore *store = (NodusLossStore*)calloc(1, sizeof(NodusLossStore));
    if (!store) return NULL;

    store_init(store, max_channels, max_records_per_channel, /*cross_process=*/0);
    return store;
}

NODUS_API void nodus_loss_store_destroy(NodusLossStore *store) {
    if (!store) return;
    mutex_destroy(&store->lock);
    free(store);
}

/* ------------------------------------------------------------------ */
/*  Global singleton backed by OS named shared memory                  */
/* ------------------------------------------------------------------ */

#define NODUS_SHM_NAME      "NodusLossStoreGlobal_v1"
#define NODUS_MTX_NAME      "NodusLossStoreMutex_v1"
#define NODUS_SHM_SIZE      sizeof(NodusLossStore)

#ifdef _WIN32

static HANDLE     g_shm_handle  = NULL;
/* g_global declared above (forward declaration) */

NODUS_API NodusLossStore* nodus_loss_store_get_global(void)
{
    if (g_global) return g_global;

    /* Create / open the cross-process named mutex. */
    g_shm_mutex = CreateMutexA(NULL, FALSE, NODUS_MTX_NAME);
    if (!g_shm_mutex) {
        fprintf(stderr, "[nodus] FATAL: CreateMutexA failed (%lu)\n",
                GetLastError());
        abort();
    }

    /* Try to open existing shared memory; if not, create new. */
    g_shm_handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, NODUS_SHM_NAME);
    int created = 0;
    if (!g_shm_handle) {
        g_shm_handle = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
            (DWORD)((unsigned long long)NODUS_SHM_SIZE >> 32),
            (DWORD)(NODUS_SHM_SIZE & 0xFFFFFFFF),
            NODUS_SHM_NAME);
        if (!g_shm_handle) {
            fprintf(stderr, "[nodus] FATAL: CreateFileMappingA failed (%lu)\n",
                    GetLastError());
            abort();
        }
        created = (GetLastError() != ERROR_ALREADY_EXISTS);
    }

    g_global = (NodusLossStore*)MapViewOfFile(
        g_shm_handle, FILE_MAP_ALL_ACCESS, 0, 0, NODUS_SHM_SIZE);
    if (!g_global) {
        fprintf(stderr, "[nodus] FATAL: MapViewOfFile failed (%lu)\n",
                GetLastError());
        abort();
    }

    /* Initialise only if we are the first creator. */
    if (created && g_global->initialized != 0x4E4F4455) {
        memset(g_global, 0, NODUS_SHM_SIZE);
        store_init(g_global, NODUS_MAX_CHANNELS, NODUS_MAX_RECORDS,
                   /*cross_process=*/1);
    }

    return g_global;
}

#else /* POSIX */

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
/* g_global declared above (forward declaration) */

NODUS_API NodusLossStore* nodus_loss_store_get_global(void)
{
    if (g_global) return g_global;

    int created = 0;
    int fd = shm_open("/" NODUS_SHM_NAME, O_RDWR, 0600);
    if (fd < 0) {
        fd = shm_open("/" NODUS_SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            perror("[nodus] FATAL: shm_open");
            abort();
        }
        if (ftruncate(fd, (off_t)NODUS_SHM_SIZE) != 0) {
            perror("[nodus] FATAL: ftruncate");
            abort();
        }
        created = 1;
    }

    g_global = (NodusLossStore*)mmap(
        NULL, NODUS_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_global == MAP_FAILED) {
        g_global = NULL;
        perror("[nodus] FATAL: mmap");
        abort();
    }

    if (created && g_global->initialized != 0x4E4F4455) {
        memset(g_global, 0, NODUS_SHM_SIZE);
        store_init(g_global, NODUS_MAX_CHANNELS, NODUS_MAX_RECORDS,
                   /*cross_process=*/1);
    }

    return g_global;
}

#endif

/* ------------------------------------------------------------------ */
/*  Write                                                             */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_loss_store_record(
        NodusLossStore *store,
        const char     *channel_key,
        float           loss,
        float           aux,
        int32_t         round_id,
        double          ts)
{
    if (!store || !channel_key) return -1;

    store_lock(store);

    int idx = find_channel(store, channel_key);
    if (idx < 0) {
        /* Allocate a new channel slot. */
        if (store->channel_count >= store->max_channels) {
            store_unlock(store);
            return -1;
        }
        idx = store->channel_count++;
        strncpy(store->channels[idx].name, channel_key,
                NODUS_MAX_CHANNEL_NAME - 1);
        store->channels[idx].name[NODUS_MAX_CHANNEL_NAME - 1] = '\0';
        store->channels[idx].length      = 0;
        store->channels[idx].step_cursor = 0;
        store->channels[idx].head        = 0;
    }

    NodusChannel *ch = &store->channels[idx];
    int32_t step = ch->step_cursor++;

    /* Determine where to write. */
    int write_pos;
    if (ch->length < ch->capacity) {
        write_pos = buf_index(ch, ch->length);
        ch->length++;
    } else {
        /* Overwrite oldest -- advance head. */
        write_pos = ch->head;
        ch->head = (ch->head + 1) % ch->capacity;
    }

    NodusLossRecord *rec = &ch->records[write_pos];
    rec->step     = step;
    rec->round_id = round_id;
    rec->loss     = isfinite(loss) ? loss : NAN;
    rec->aux      = aux;
    rec->ts       = ts > 0.0 ? ts : 0.0;

    store_unlock(store);
    return step;
}

NODUS_API void nodus_loss_store_clear(NodusLossStore *store) {
    if (!store) return;
    store_lock(store);
    for (int i = 0; i < store->channel_count; i++) {
        store->channels[i].length      = 0;
        store->channels[i].step_cursor = 0;
        store->channels[i].head        = 0;
        store->channels[i].name[0]     = '\0';
    }
    store->channel_count = 0;
    store_unlock(store);
}

NODUS_API int nodus_loss_store_clear_channel(
        NodusLossStore *store, const char *channel_key)
{
    if (!store || !channel_key) return -1;
    store_lock(store);
    int idx = find_channel(store, channel_key);
    if (idx < 0) {
        store_unlock(store);
        return -1;
    }
    NodusChannel *ch = &store->channels[idx];
    ch->length      = 0;
    ch->step_cursor = 0;
    ch->head        = 0;
    /* Compact: move last channel into this slot if not already last. */
    int last = store->channel_count - 1;
    if (idx != last) {
        memcpy(&store->channels[idx], &store->channels[last],
               sizeof(NodusChannel));
    }
    store->channels[last].name[0] = '\0';
    store->channels[last].length  = 0;
    store->channel_count--;
    store_unlock(store);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Read -- channel enumeration                                        */
/* ------------------------------------------------------------------ */

NODUS_API int nodus_loss_store_channel_count(const NodusLossStore *store) {
    int count;
    NodusLossStore *st = (NodusLossStore*)store;
    if (!st) return 0;
    store_lock(st);
    count = st->channel_count;
    store_unlock(st);
    return count;
}

NODUS_API int nodus_loss_store_channel_name(
        const NodusLossStore *store, int index, char *buf, int buf_len)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int len, copy;
    if (!st || !buf || buf_len <= 0) return -1;
    store_lock(st);
    if (index < 0 || index >= st->channel_count) {
        store_unlock(st);
        return -1;
    }
    len = (int)strlen(st->channels[index].name);
    copy = (len < buf_len - 1) ? len : (buf_len - 1);
    memcpy(buf, st->channels[index].name, (size_t)copy);
    buf[copy] = '\0';
    store_unlock(st);
    return len;
}

/* ------------------------------------------------------------------ */
/*  Read -- per-channel queries                                        */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_loss_store_channel_length(
        const NodusLossStore *store, const char *channel_key)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    int32_t length;
    if (!st || !channel_key) return 0;
    store_lock(st);
    idx = find_channel(st, channel_key);
    length = (idx >= 0) ? st->channels[idx].length : 0;
    store_unlock(st);
    return length;
}

NODUS_API int32_t nodus_loss_store_channel_cursor(
        const NodusLossStore *store, const char *channel_key)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    int32_t cursor;
    if (!st || !channel_key) return 0;
    store_lock(st);
    idx = find_channel(st, channel_key);
    cursor = (idx >= 0) ? st->channels[idx].step_cursor : 0;
    store_unlock(st);
    return cursor;
}

NODUS_API int32_t nodus_loss_store_query_since(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        NodusLossRecord      *out_buf,
        int32_t               max_out)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    int32_t copied = 0;
    int32_t i;
    if (!st || !channel_key || !out_buf || max_out <= 0) return -1;

    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) {
        store_unlock(st);
        return -1;
    }

    ch = &st->channels[idx];

    for (i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step) {
            out_buf[copied++] = ch->records[bi];
        }
    }
    store_unlock(st);
    return copied;
}

NODUS_API int nodus_loss_store_latest(
        const NodusLossStore *store,
        const char           *channel_key,
        NodusLossRecord      *out)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx, bi;
    const NodusChannel *ch;
    if (!st || !channel_key || !out) return -1;

    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) {
        store_unlock(st);
        return -1;
    }

    ch = &st->channels[idx];
    if (ch->length == 0) {
        store_unlock(st);
        return -1;
    }

    /* Latest is at logical position length-1. */
    bi = buf_index(ch, ch->length - 1);
    *out = ch->records[bi];
    store_unlock(st);
    return 0;
}

NODUS_API int nodus_loss_store_channel_data_ptr(
        const NodusLossStore  *store,
        const char            *channel_key,
        const NodusLossRecord **out_ptr,
        int32_t               *out_length)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    if (!st || !channel_key || !out_ptr || !out_length) return -1;

    /* NOTE: Caller MUST hold the external lock to use the returned pointer
       safely.  This function acquires the lock only to resolve the channel
       and copy the pointer/length; the data behind the pointer can change
       as soon as the lock is released. */
    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) {
        store_unlock(st);
        return -1;
    }

    ch = &st->channels[idx];
    *out_ptr   = ch->records;
    *out_length = ch->length;
    store_unlock(st);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Locking                                                           */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_loss_store_lock(NodusLossStore *store) {
    if (store) store_lock(store);
}

NODUS_API void nodus_loss_store_unlock(NodusLossStore *store) {
    if (store) store_unlock(store);
}

/* ------------------------------------------------------------------ */
/*  Bulk field extractors (flat arrays for tensor wrapping)           */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_loss_store_get_loss_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        float                *out_buf,
        int32_t               max_out)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    int32_t copied = 0;
    if (!st || !channel_key || !out_buf || max_out <= 0) return -1;
    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) { store_unlock(st); return -1; }
    ch = &st->channels[idx];
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].loss;
    }
    store_unlock(st);
    return copied;
}

NODUS_API int32_t nodus_loss_store_get_ts_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        double               *out_buf,
        int32_t               max_out)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    int32_t copied = 0;
    if (!st || !channel_key || !out_buf || max_out <= 0) return -1;
    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) { store_unlock(st); return -1; }
    ch = &st->channels[idx];
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].ts;
    }
    store_unlock(st);
    return copied;
}

NODUS_API int32_t nodus_loss_store_get_step_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        int32_t              *out_buf,
        int32_t               max_out)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    int32_t copied = 0;
    if (!st || !channel_key || !out_buf || max_out <= 0) return -1;
    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) { store_unlock(st); return -1; }
    ch = &st->channels[idx];
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].step;
    }
    store_unlock(st);
    return copied;
}

/* ------------------------------------------------------------------ */
/*  Y-range and time-range helpers                                    */
/* ------------------------------------------------------------------ */

NODUS_API int nodus_loss_store_channel_y_range(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        float                *out_min,
        float                *out_max)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx, count = 0;
    const NodusChannel *ch;
    float lo, hi, span;
    if (!st || !channel_key || !out_min || !out_max) return -1;

    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) { store_unlock(st); return -1; }
    ch = &st->channels[idx];
    lo =  1e30f;
    hi = -1e30f;
    for (int32_t i = 0; i < ch->length; i++) {
        int bi = buf_index(ch, i);
        const NodusLossRecord *r = &ch->records[bi];
        if (r->step < from_step) continue;
        if (!isfinite(r->loss)) continue;
        if (r->loss < lo) lo = r->loss;
        if (r->loss > hi) hi = r->loss;
        count++;
    }
    store_unlock(st);
    if (count < 1) return -1;
    span = hi - lo;
    *out_min = (lo - span * 0.05f > 0.0f) ? lo - span * 0.05f : 0.0f;
    *out_max = hi + span * 0.05f;
    if (*out_max <= *out_min + 1e-9f) *out_max = *out_min + 1.0f;
    return 0;
}

NODUS_API int nodus_loss_store_global_time_range(
        const NodusLossStore *store,
        double               *out_t_min,
        double               *out_t_max)
{
    NodusLossStore *st = (NodusLossStore*)store;
    double lo, hi;
    int found = 0;
    if (!st || !out_t_min || !out_t_max) return -1;

    store_lock(st);
    lo = 1e30; hi = -1e30;
    for (int c = 0; c < st->channel_count; c++) {
        const NodusChannel *ch = &st->channels[c];
        for (int32_t i = 0; i < ch->length; i++) {
            int bi = buf_index(ch, i);
            double t = ch->records[bi].ts;
            if (t > 0.0) {
                if (t < lo) lo = t;
                if (t > hi) hi = t;
                found = 1;
            }
        }
    }
    store_unlock(st);
    if (!found) return -1;
    *out_t_min = lo;
    *out_t_max = hi;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Graph-line renderer                                               */
/* ------------------------------------------------------------------ */

/* Bresenham-style thick anti-aliased line between two points on RGBA buf. */
static void draw_line_rgba(uint8_t *buf, int W, int H,
                           int x0, int y0, int x1, int y1,
                           uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    /* Clip trivially out-of-bounds lines. */
    if ((x0 < 0 && x1 < 0) || (x0 >= W && x1 >= W)) return;
    if ((y0 < 0 && y1 < 0) || (y0 >= H && y1 >= H)) return;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            int off = (y0 * W + x0) * 4;
            buf[off + 0] = r;
            buf[off + 1] = g;
            buf[off + 2] = b;
            buf[off + 3] = a;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

NODUS_API int32_t nodus_loss_store_render_graph_line(
        const NodusLossStore       *store,
        const char                 *channel_key,
        const NodusGraphLineConfig *cfg,
        uint8_t                    *out_rgba)
{
    NodusLossStore *st = (NodusLossStore*)store;
    int idx;
    const NodusChannel *ch;
    int32_t pw, ph, from_step, n_visible, vi, plotted;
    float ymin, ymax;
    double tmin, tmax, t_span;
    int use_time, use_heap;
    int *px_buf = NULL;
    int stack_buf[8192 * 2];

    if (!st || !channel_key || !cfg || !out_rgba) return -1;

    store_lock(st);
    idx = find_channel(st, channel_key);
    if (idx < 0) { store_unlock(st); return -1; }

    ch = &st->channels[idx];
    pw = cfg->plot_w;
    ph = cfg->plot_h;
    ymin = cfg->y_min;
    ymax = cfg->y_max;
    tmin = cfg->t_min;
    tmax = cfg->t_max;
    use_time = cfg->use_time_axis;
    from_step = cfg->from_step;

    if (pw <= 0 || ph <= 0) { store_unlock(st); return -1; }
    if (ymax <= ymin + 1e-9f) { store_unlock(st); return -1; }

    /* Collect visible records. */
    n_visible = 0;
    for (int32_t i = 0; i < ch->length; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step) n_visible++;
    }
    if (n_visible < 2) { store_unlock(st); return 0; }

    use_heap = (n_visible > 8192);
    if (use_heap) {
        px_buf = (int*)malloc(sizeof(int) * (size_t)n_visible * 2);
        if (!px_buf) { store_unlock(st); return -1; }
    } else {
        px_buf = stack_buf;
    }

    /* Compute time span for step-index mode. */
    t_span = (tmax > tmin + 1e-9) ? (tmax - tmin) : 1.0;
    vi = 0;
    plotted = 0;
    for (int32_t i = 0; i < ch->length; i++) {
        int bi = buf_index(ch, i);
        const NodusLossRecord *r = &ch->records[bi];
        if (r->step < from_step) continue;
        if (!isfinite(r->loss)) { vi++; continue; }

        int xp, yp;
        if (use_time && r->ts > 0.0) {
            double tfrac = (r->ts - tmin) / t_span;
            xp = (int)(tfrac * (double)(pw - 1));
        } else {
            xp = (int)((double)vi / (double)(n_visible - 1) * (double)(pw - 1));
        }
        float yfrac = (r->loss - ymin) / (ymax - ymin);
        yp = (int)((double)(ph - 1) * (1.0 - (double)yfrac));

        if (xp < 0) xp = 0; if (xp >= pw) xp = pw - 1;
        if (yp < 0) yp = 0; if (yp >= ph) yp = ph - 1;

        px_buf[plotted * 2 + 0] = xp;
        px_buf[plotted * 2 + 1] = yp;
        plotted++;
        vi++;
    }
    store_unlock(st);

    /* Draw line segments (operates on local px_buf, no lock needed). */
    for (int32_t j = 1; j < plotted; j++) {
        draw_line_rgba(out_rgba, pw, ph,
                       px_buf[(j-1)*2], px_buf[(j-1)*2+1],
                       px_buf[j*2],     px_buf[j*2+1],
                       cfg->r, cfg->g, cfg->b, cfg->a);
    }

    if (use_heap) free(px_buf);
    return plotted;
}

/* ------------------------------------------------------------------ */
/*  Batch graph renderer                                              */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_loss_store_render_all_lines(
        const NodusLossStore *store,
        int32_t               plot_w,
        int32_t               plot_h,
        float                 display_start_frac,
        int32_t               num_channels,
        const char *const    *channel_keys,
        const uint8_t        *colors,
        uint8_t              *out_rgba,
        float                *out_y_min,
        float                *out_y_max,
        double               *out_t_min,
        double               *out_t_max)
{
    if (!store || !channel_keys || !colors || !out_rgba) return -1;
    if (!out_y_min || !out_y_max || !out_t_min || !out_t_max) return -1;
    if (plot_w <= 0 || plot_h <= 0 || num_channels <= 0) return -1;

    float sf = display_start_frac;
    if (sf < 0.0f) sf = 0.0f;
    if (sf > 0.99f) sf = 0.99f;

    store_lock((NodusLossStore*)store);

    /* --- Pass 1: compute global y_range and t_range across requested channels --- */
    float y_lo =  1e30f, y_hi = -1e30f;
    double t_lo = 1e30, t_hi = -1e30;
    int any_data = 0;

    for (int32_t c = 0; c < num_channels; c++) {
        int idx = find_channel(store, channel_keys[c]);
        if (idx < 0) continue;
        const NodusChannel *ch = &store->channels[idx];
        int32_t i_start = (int32_t)(sf * (float)ch->length);
        if (i_start < 0) i_start = 0;
        for (int32_t i = i_start; i < ch->length; i++) {
            int bi = buf_index(ch, i);
            const NodusLossRecord *r = &ch->records[bi];
            if (isfinite(r->loss) && r->loss >= 0.0f) {
                if (r->loss < y_lo) y_lo = r->loss;
                if (r->loss > y_hi) y_hi = r->loss;
                any_data = 1;
            }
            if (r->ts > 0.0) {
                if (r->ts < t_lo) t_lo = r->ts;
                if (r->ts > t_hi) t_hi = r->ts;
            }
        }
    }

    if (!any_data) {
        store_unlock((NodusLossStore*)store);
        *out_y_min = 0.0f; *out_y_max = 1.0f;
        *out_t_min = 0.0;  *out_t_max = 0.0;
        return 0;
    }

    /* Apply 5% padding to y range. */
    float y_span = y_hi - y_lo;
    float ymin = (y_lo - y_span * 0.05f > 0.0f) ? y_lo - y_span * 0.05f : 0.0f;
    float ymax = y_hi + y_span * 0.05f;
    if (ymax <= ymin + 1e-9f) ymax = ymin + 1.0f;

    int has_time = (t_hi > t_lo + 1e-9);
    double t_span_d = has_time ? (t_hi - t_lo) : 1.0;

    *out_y_min = ymin;
    *out_y_max = ymax;
    *out_t_min = has_time ? t_lo : 0.0;
    *out_t_max = has_time ? t_hi : 0.0;

    /* --- Pass 2: render each channel onto the shared buffer --- */
    int32_t total_segments = 0;

    for (int32_t c = 0; c < num_channels; c++) {
        int idx = find_channel(store, channel_keys[c]);
        if (idx < 0) continue;
        const NodusChannel *ch = &store->channels[idx];
        int32_t i_start = (int32_t)(sf * (float)ch->length);
        if (i_start < 0) i_start = 0;
        int32_t n_visible = ch->length - i_start;
        if (n_visible < 2) continue;

        uint8_t cr = colors[c * 4 + 0];
        uint8_t cg = colors[c * 4 + 1];
        uint8_t cb = colors[c * 4 + 2];
        uint8_t ca = colors[c * 4 + 3];

        /* Allocate pixel coordinate buffer. */
        int use_heap = (n_visible > 8192);
        int *px_buf = NULL;
        int stack_buf[8192 * 2];
        if (use_heap) {
            px_buf = (int*)malloc(sizeof(int) * (size_t)n_visible * 2);
            if (!px_buf) continue;
        } else {
            px_buf = stack_buf;
        }

        int32_t plotted = 0;
        int32_t vi = 0;
        for (int32_t i = i_start; i < ch->length; i++) {
            int bi = buf_index(ch, i);
            const NodusLossRecord *r = &ch->records[bi];
            if (!isfinite(r->loss)) { vi++; continue; }

            int xp, yp;
            if (has_time && r->ts > 0.0) {
                double tfrac = (r->ts - t_lo) / t_span_d;
                xp = (int)(tfrac * (double)(plot_w - 1));
            } else {
                xp = (int)((double)vi / (double)(n_visible - 1) * (double)(plot_w - 1));
            }
            float yfrac = (r->loss - ymin) / (ymax - ymin);
            yp = (int)((double)(plot_h - 1) * (1.0 - (double)yfrac));

            if (xp < 0) xp = 0; if (xp >= plot_w) xp = plot_w - 1;
            if (yp < 0) yp = 0; if (yp >= plot_h) yp = plot_h - 1;

            px_buf[plotted * 2 + 0] = xp;
            px_buf[plotted * 2 + 1] = yp;
            plotted++;
            vi++;
        }

        for (int32_t j = 1; j < plotted; j++) {
            draw_line_rgba(out_rgba, plot_w, plot_h,
                           px_buf[(j-1)*2], px_buf[(j-1)*2+1],
                           px_buf[j*2],     px_buf[j*2+1],
                           cr, cg, cb, ca);
        }
        total_segments += (plotted > 0) ? plotted - 1 : 0;

        if (use_heap) free(px_buf);
    }

    store_unlock((NodusLossStore*)store);
    return total_segments;
}

/* ================================================================== */
/*  Scrub Ring -- cross-process training-image & frame cache          */
/* ================================================================== */

typedef struct NodusScrubEntry {
    int32_t  step;
    int32_t  round_id;
    double   ts;
    float    loss;
    char     channel_key[NODUS_MAX_CHANNEL_NAME];
    uint32_t flags;
    uint32_t image_w;
    uint32_t image_h;
    uint32_t output_w;      /* current output image dims (may shrink via reduce) */
    uint32_t output_h;
    uint32_t target_len;
    uint32_t _pad0;
    uint8_t  training_image[NODUS_SCRUB_IMAGE_BYTES];
    uint8_t  output_image[NODUS_SCRUB_IMAGE_BYTES];
    uint8_t  target_data[NODUS_SCRUB_TARGET_BYTES];
    uint8_t  thumbnails[NODUS_SCRUB_NUM_THUMBS][NODUS_SCRUB_THUMB_BYTES];
    /* Frame text — parallel to images, written under ring lock. */
    char     frame_caption[NODUS_SCRUB_CAPTION_BYTES];
    char     frame_titles[NODUS_SCRUB_NUM_PANELS][NODUS_SCRUB_TITLE_BYTES];
    char     frame_rows[NODUS_SCRUB_NUM_PANELS][NODUS_SCRUB_ROWS_BYTES];
} NodusScrubEntry;

struct NodusScrubRing {
    nodus_mutex_t lock;
    int32_t capacity;
    int32_t length;
    int32_t head;
    int32_t write_cursor;
    int32_t initialized;    /* 0x53435242 = 'SCRB' */
    int32_t _pad[3];
    NodusScrubEntry entries[NODUS_SCRUB_RING_CAPACITY];
};

#define NODUS_SCRUB_SHM_NAME    "NodusScrubRingGlobal_v2"
#define NODUS_SCRUB_MTX_NAME    "NodusScrubRingMutex_v2"
#define NODUS_SCRUB_SHM_SIZE    sizeof(NodusScrubRing)

static NodusScrubRing *g_scrub_global = NULL;

#ifdef _WIN32
static HANDLE g_scrub_shm_handle = NULL;
static HANDLE g_scrub_shm_mutex  = NULL;

static void ring_lock(NodusScrubRing *ring) {
    if (ring == g_scrub_global && g_scrub_shm_mutex)
        WaitForSingleObject(g_scrub_shm_mutex, INFINITE);
    else
        EnterCriticalSection(&ring->lock);
}
static void ring_unlock(NodusScrubRing *ring) {
    if (ring == g_scrub_global && g_scrub_shm_mutex)
        ReleaseMutex(g_scrub_shm_mutex);
    else
        LeaveCriticalSection(&ring->lock);
}
#else
static void ring_lock(NodusScrubRing *ring)   { pthread_mutex_lock(&ring->lock); }
static void ring_unlock(NodusScrubRing *ring) { pthread_mutex_unlock(&ring->lock); }
#endif

static int scrub_buf_index(const NodusScrubRing *ring, int logical) {
    return (ring->head + logical) % ring->capacity;
}

static void ring_init(NodusScrubRing *ring, int cross_process)
{
#ifdef _WIN32
    (void)cross_process;
    mutex_init(&ring->lock);
#else
    if (cross_process)
        mutex_init_shared(&ring->lock);
    else
        mutex_init(&ring->lock);
#endif
    ring->capacity     = NODUS_SCRUB_RING_CAPACITY;
    ring->length       = 0;
    ring->head         = 0;
    ring->write_cursor = 0;
    ring->initialized  = 0x53435242; /* 'SCRB' */
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring global singleton (separate shared memory segment)      */
/* ------------------------------------------------------------------ */

#ifdef _WIN32

NODUS_API NodusScrubRing* nodus_scrub_ring_get_global(void)
{
    if (g_scrub_global) return g_scrub_global;

    g_scrub_shm_mutex = CreateMutexA(NULL, FALSE, NODUS_SCRUB_MTX_NAME);
    if (!g_scrub_shm_mutex) {
        fprintf(stderr, "[nodus] FATAL: CreateMutexA scrub ring failed (%lu)\n",
                GetLastError());
        abort();
    }

    g_scrub_shm_handle = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS, FALSE, NODUS_SCRUB_SHM_NAME);
    int created = 0;
    if (!g_scrub_shm_handle) {
        DWORD size_hi = (DWORD)((unsigned long long)NODUS_SCRUB_SHM_SIZE >> 32);
        DWORD size_lo = (DWORD)(NODUS_SCRUB_SHM_SIZE & 0xFFFFFFFF);
        g_scrub_shm_handle = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
            size_hi, size_lo, NODUS_SCRUB_SHM_NAME);
        if (!g_scrub_shm_handle) {
            fprintf(stderr,
                    "[nodus] FATAL: CreateFileMappingA scrub ring failed (%lu)\n",
                    GetLastError());
            abort();
        }
        created = (GetLastError() != ERROR_ALREADY_EXISTS);
    }

    g_scrub_global = (NodusScrubRing*)MapViewOfFile(
        g_scrub_shm_handle, FILE_MAP_ALL_ACCESS, 0, 0, NODUS_SCRUB_SHM_SIZE);
    if (!g_scrub_global) {
        fprintf(stderr,
                "[nodus] FATAL: MapViewOfFile scrub ring failed (%lu)\n",
                GetLastError());
        abort();
    }

    if (created && g_scrub_global->initialized != 0x53435242) {
        memset(g_scrub_global, 0, NODUS_SCRUB_SHM_SIZE);
        ring_init(g_scrub_global, /*cross_process=*/1);
    }

    return g_scrub_global;
}

#else /* POSIX */

NODUS_API NodusScrubRing* nodus_scrub_ring_get_global(void)
{
    if (g_scrub_global) return g_scrub_global;

    int created = 0;
    int fd = shm_open("/" NODUS_SCRUB_SHM_NAME, O_RDWR, 0600);
    if (fd < 0) {
        fd = shm_open("/" NODUS_SCRUB_SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            perror("[nodus] FATAL: shm_open scrub ring");
            abort();
        }
        if (ftruncate(fd, (off_t)NODUS_SCRUB_SHM_SIZE) != 0) {
            perror("[nodus] FATAL: ftruncate scrub ring");
            abort();
        }
        created = 1;
    }

    g_scrub_global = (NodusScrubRing*)mmap(
        NULL, NODUS_SCRUB_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_scrub_global == MAP_FAILED) {
        g_scrub_global = NULL;
        perror("[nodus] FATAL: mmap scrub ring");
        abort();
    }

    if (created && g_scrub_global->initialized != 0x53435242) {
        memset(g_scrub_global, 0, NODUS_SCRUB_SHM_SIZE);
        ring_init(g_scrub_global, /*cross_process=*/1);
    }

    return g_scrub_global;
}

#endif

/* ------------------------------------------------------------------ */
/*  Scrub Ring create / destroy (heap-allocated, single-process)      */
/* ------------------------------------------------------------------ */

NODUS_API NodusScrubRing* nodus_scrub_ring_create(void)
{
    NodusScrubRing *ring = (NodusScrubRing*)calloc(1, sizeof(NodusScrubRing));
    if (!ring) return NULL;
    ring_init(ring, /*cross_process=*/0);
    return ring;
}

NODUS_API void nodus_scrub_ring_destroy(NodusScrubRing *ring)
{
    if (!ring) return;
    mutex_destroy(&ring->lock);
    free(ring);
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring write                                                  */
/* ------------------------------------------------------------------ */

/* Expand RGB (3 bytes/pixel) to RGBA (4 bytes/pixel), alpha = 255.
   src must contain w*h*3 bytes; dst must hold w*h*4 bytes. */
static void expand_rgb3_to_rgba4(const uint8_t *src, uint8_t *dst,
                                 uint32_t w, uint32_t h)
{
    uint32_t npix = w * h;
    for (uint32_t i = 0; i < npix; i++) {
        dst[i * 4 + 0] = src[i * 3 + 0];
        dst[i * 4 + 1] = src[i * 3 + 1];
        dst[i * 4 + 2] = src[i * 3 + 2];
        dst[i * 4 + 3] = 255;
    }
}

NODUS_API int32_t nodus_scrub_ring_push(
        NodusScrubRing *ring,
        int32_t step, int32_t round_id, double ts,
        float loss, const char *channel_key, uint32_t flags,
        uint32_t image_w, uint32_t image_h,
        const uint8_t *training_image, uint32_t training_image_len,
        const uint8_t *output_image, uint32_t output_image_len,
        const uint8_t *target_data, uint32_t target_len,
        const uint8_t *thumb0,
        const uint8_t *thumb1,
        const uint8_t *thumb2)
{
    if (!ring) return -1;

    ring_lock(ring);

    int write_pos;
    if (ring->length < ring->capacity) {
        write_pos = scrub_buf_index(ring, ring->length);
        ring->length++;
    } else {
        write_pos = ring->head;
        ring->head = (ring->head + 1) % ring->capacity;
    }

    NodusScrubEntry *e = &ring->entries[write_pos];
    e->step      = step;
    e->round_id  = round_id;
    e->ts        = ts;
    e->loss      = loss;
    e->flags     = flags;
    e->image_w   = image_w;
    e->image_h   = image_h;
    e->output_w  = image_w;
    e->output_h  = image_h;
    e->target_len = 0;
    e->_pad0     = 0;

    if (channel_key) {
        strncpy(e->channel_key, channel_key, NODUS_MAX_CHANNEL_NAME - 1);
        e->channel_key[NODUS_MAX_CHANNEL_NAME - 1] = '\0';
    } else {
        e->channel_key[0] = '\0';
    }

    /* Training image (input).  Auto-expand RGB→RGBA when len == w*h*3. */
    if (training_image && training_image_len > 0) {
        uint32_t rgb3 = image_w * image_h * 3;
        uint32_t rgba4 = image_w * image_h * 4;
        if (training_image_len == rgb3 && rgba4 <= NODUS_SCRUB_IMAGE_BYTES && rgb3 > 0) {
            expand_rgb3_to_rgba4(training_image, e->training_image, image_w, image_h);
        } else {
            uint32_t n = training_image_len;
            if (n > NODUS_SCRUB_IMAGE_BYTES) n = NODUS_SCRUB_IMAGE_BYTES;
            memcpy(e->training_image, training_image, n);
        }
        e->flags |= NODUS_SCRUB_FLAG_HAS_IMAGE;
    }

    /* Output image (network output).  Auto-expand RGB→RGBA when len == w*h*3. */
    if (output_image && output_image_len > 0) {
        uint32_t rgb3 = image_w * image_h * 3;
        uint32_t rgba4 = image_w * image_h * 4;
        if (output_image_len == rgb3 && rgba4 <= NODUS_SCRUB_IMAGE_BYTES && rgb3 > 0) {
            expand_rgb3_to_rgba4(output_image, e->output_image, image_w, image_h);
        } else {
            uint32_t n = output_image_len;
            if (n > NODUS_SCRUB_IMAGE_BYTES) n = NODUS_SCRUB_IMAGE_BYTES;
            memcpy(e->output_image, output_image, n);
        }
        e->flags |= NODUS_SCRUB_FLAG_HAS_OUTPUT;
    }

    /* Target / mask data.  Auto-expand RGB→RGBA when len == w*h*3. */
    if (target_data && target_len > 0) {
        uint32_t rgb3 = image_w * image_h * 3;
        uint32_t rgba4 = image_w * image_h * 4;
        if (target_len == rgb3 && rgba4 <= NODUS_SCRUB_TARGET_BYTES && rgb3 > 0) {
            expand_rgb3_to_rgba4(target_data, e->target_data, image_w, image_h);
            e->target_len = rgba4;
        } else {
            uint32_t n = target_len;
            if (n > NODUS_SCRUB_TARGET_BYTES) n = NODUS_SCRUB_TARGET_BYTES;
            memcpy(e->target_data, target_data, n);
            e->target_len = n;
        }
        e->flags |= NODUS_SCRUB_FLAG_HAS_TARGET;
    }

    /* Thumbnails (optional, GUI may produce these itself). */
    if (thumb0) {
        memcpy(e->thumbnails[0], thumb0, NODUS_SCRUB_THUMB_BYTES);
        e->flags |= NODUS_SCRUB_FLAG_HAS_THUMBS;
    }
    if (thumb1) {
        memcpy(e->thumbnails[1], thumb1, NODUS_SCRUB_THUMB_BYTES);
        e->flags |= NODUS_SCRUB_FLAG_HAS_THUMBS;
    }
    if (thumb2) {
        memcpy(e->thumbnails[2], thumb2, NODUS_SCRUB_THUMB_BYTES);
        e->flags |= NODUS_SCRUB_FLAG_HAS_THUMBS;
    }

    int32_t cursor = ring->write_cursor++;

    ring_unlock(ring);
    return cursor;
}

NODUS_API void nodus_scrub_ring_clear(NodusScrubRing *ring)
{
    if (!ring) return;
    ring_lock(ring);
    ring->length       = 0;
    ring->head         = 0;
    ring->write_cursor = 0;
    ring_unlock(ring);
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring read                                                   */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_scrub_ring_length(const NodusScrubRing *ring) {
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int32_t len;
    if (!r) return 0;
    ring_lock(r);
    len = r->length;
    ring_unlock(r);
    return len;
}

NODUS_API int32_t nodus_scrub_ring_capacity(const NodusScrubRing *ring) {
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int32_t cap;
    if (!r) return 0;
    ring_lock(r);
    cap = r->capacity;
    ring_unlock(r);
    return cap;
}

NODUS_API int32_t nodus_scrub_ring_write_cursor(const NodusScrubRing *ring) {
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int32_t cur;
    if (!r) return 0;
    ring_lock(r);
    cur = r->write_cursor;
    ring_unlock(r);
    return cur;
}

NODUS_API int nodus_scrub_ring_get_meta(
        const NodusScrubRing *ring, int32_t index,
        int32_t *out_step, int32_t *out_round_id, double *out_ts,
        float *out_loss, char *out_channel_key, int channel_key_buflen,
        uint32_t *out_flags,
        uint32_t *out_image_w, uint32_t *out_image_h,
        uint32_t *out_target_len,
        uint32_t *out_output_w, uint32_t *out_output_h)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    if (!r) return -1;

    ring_lock(r);
    if (index < 0 || index >= r->length) {
        ring_unlock(r);
        return -1;
    }

    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];

    if (out_step)      *out_step      = e->step;
    if (out_round_id)  *out_round_id  = e->round_id;
    if (out_ts)        *out_ts        = e->ts;
    if (out_loss)      *out_loss      = e->loss;
    if (out_flags)     *out_flags     = e->flags;
    if (out_image_w)   *out_image_w   = e->image_w;
    if (out_image_h)   *out_image_h   = e->image_h;
    if (out_target_len) *out_target_len = e->target_len;
    if (out_output_w)  *out_output_w  = e->output_w;
    if (out_output_h)  *out_output_h  = e->output_h;

    if (out_channel_key && channel_key_buflen > 0) {
        strncpy(out_channel_key, e->channel_key, channel_key_buflen - 1);
        out_channel_key[channel_key_buflen - 1] = '\0';
    }
    ring_unlock(r);

    return 0;
}

NODUS_API int32_t nodus_scrub_ring_copy_output_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    uint32_t img_bytes, n;
    if (!r || !out_buf) return -1;

    ring_lock(r);
    if (index < 0 || index >= r->length) { ring_unlock(r); return -1; }
    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_OUTPUT)) { ring_unlock(r); return 0; }
    img_bytes = e->output_w * e->output_h * NODUS_SCRUB_IMAGE_C;
    if (img_bytes > NODUS_SCRUB_IMAGE_BYTES) img_bytes = NODUS_SCRUB_IMAGE_BYTES;
    n = (img_bytes < buf_size) ? img_bytes : buf_size;
    memcpy(out_buf, e->output_image, n);
    ring_unlock(r);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_training_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    uint32_t img_bytes, n;
    if (!r || !out_buf) return -1;

    ring_lock(r);
    if (index < 0 || index >= r->length) { ring_unlock(r); return -1; }
    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_IMAGE)) { ring_unlock(r); return 0; }
    img_bytes = e->image_w * e->image_h * NODUS_SCRUB_IMAGE_C;
    if (img_bytes > NODUS_SCRUB_IMAGE_BYTES) img_bytes = NODUS_SCRUB_IMAGE_BYTES;
    n = (img_bytes < buf_size) ? img_bytes : buf_size;
    memcpy(out_buf, e->training_image, n);
    ring_unlock(r);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_target(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    uint32_t n;
    if (!r || !out_buf) return -1;

    ring_lock(r);
    if (index < 0 || index >= r->length) { ring_unlock(r); return -1; }
    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_TARGET)) { ring_unlock(r); return 0; }
    n = (e->target_len < buf_size) ? e->target_len : buf_size;
    memcpy(out_buf, e->target_data, n);
    ring_unlock(r);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_thumbnail(
        const NodusScrubRing *ring, int32_t index, int thumb_idx,
        uint8_t *out_buf, uint32_t buf_size)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    uint32_t n;
    if (!r || !out_buf) return -1;
    if (thumb_idx < 0 || thumb_idx >= NODUS_SCRUB_NUM_THUMBS) return -1;

    ring_lock(r);
    if (index < 0 || index >= r->length) { ring_unlock(r); return -1; }
    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_THUMBS)) { ring_unlock(r); return 0; }
    n = (NODUS_SCRUB_THUMB_BYTES < buf_size)
               ? NODUS_SCRUB_THUMB_BYTES : buf_size;
    memcpy(out_buf, e->thumbnails[thumb_idx], n);
    ring_unlock(r);
    return (int32_t)n;
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring text — parallel to images                             */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_scrub_ring_write_text(
        NodusScrubRing *ring, int32_t cursor,
        const char *caption,
        const char *title0, const char *title1, const char *title2,
        const char *rows0,  const char *rows1,  const char *rows2)
{
    if (!ring || cursor < 0) return;
    int slot = (int)(cursor % ring->capacity);
    ring_lock(ring);
    NodusScrubEntry *e = &ring->entries[slot];

#define _NODUS_COPY_STR(dst, src, cap) \
    if (src) { strncpy(dst, src, (cap) - 1); dst[(cap) - 1] = '\0'; } \
    else { dst[0] = '\0'; }

    _NODUS_COPY_STR(e->frame_caption,   caption, NODUS_SCRUB_CAPTION_BYTES)
    _NODUS_COPY_STR(e->frame_titles[0], title0,  NODUS_SCRUB_TITLE_BYTES)
    _NODUS_COPY_STR(e->frame_titles[1], title1,  NODUS_SCRUB_TITLE_BYTES)
    _NODUS_COPY_STR(e->frame_titles[2], title2,  NODUS_SCRUB_TITLE_BYTES)
    _NODUS_COPY_STR(e->frame_rows[0],   rows0,   NODUS_SCRUB_ROWS_BYTES)
    _NODUS_COPY_STR(e->frame_rows[1],   rows1,   NODUS_SCRUB_ROWS_BYTES)
    _NODUS_COPY_STR(e->frame_rows[2],   rows2,   NODUS_SCRUB_ROWS_BYTES)

#undef _NODUS_COPY_STR

    e->flags |= NODUS_SCRUB_FLAG_HAS_TEXT;
    ring_unlock(ring);
}

NODUS_API int nodus_scrub_ring_read_text(
        const NodusScrubRing *ring, int32_t cursor,
        char *out_caption,  int caption_buf_size,
        char *out_title0,   int title0_buf_size,
        char *out_title1,   int title1_buf_size,
        char *out_title2,   int title2_buf_size,
        char *out_rows0,    int rows0_buf_size,
        char *out_rows1,    int rows1_buf_size,
        char *out_rows2,    int rows2_buf_size)
{
    if (!ring || cursor < 0) return -1;
    int slot = (int)(cursor % ring->capacity);
    ring_lock((NodusScrubRing*)ring);
    const NodusScrubEntry *e = &ring->entries[slot];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_TEXT)) {
        ring_unlock((NodusScrubRing*)ring);
        return 0;
    }

#define _NODUS_READ_STR(out, bufsz, src) \
    if (out && (bufsz) > 0) { strncpy(out, src, (bufsz) - 1); out[(bufsz) - 1] = '\0'; }

    _NODUS_READ_STR(out_caption, caption_buf_size, e->frame_caption)
    _NODUS_READ_STR(out_title0,  title0_buf_size,  e->frame_titles[0])
    _NODUS_READ_STR(out_title1,  title1_buf_size,  e->frame_titles[1])
    _NODUS_READ_STR(out_title2,  title2_buf_size,  e->frame_titles[2])
    _NODUS_READ_STR(out_rows0,   rows0_buf_size,   e->frame_rows[0])
    _NODUS_READ_STR(out_rows1,   rows1_buf_size,   e->frame_rows[1])
    _NODUS_READ_STR(out_rows2,   rows2_buf_size,   e->frame_rows[2])

#undef _NODUS_READ_STR

    ring_unlock((NodusScrubRing*)ring);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring locking                                                */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Scrub Ring quality reduction (GUI-side cache management)          */
/* ------------------------------------------------------------------ */

/* Box-average downsample src (sw x sh x C) -> dst (tw x th x C) in-place.
   src and dst may alias (writes to dst first).  C = NODUS_SCRUB_IMAGE_C. */
static void downsample_box_rgba(
        uint8_t *buf,
        uint32_t sw, uint32_t sh,
        uint32_t tw, uint32_t th)
{
    if (tw == 0 || th == 0 || sw == 0 || sh == 0) return;
    uint32_t C = NODUS_SCRUB_IMAGE_C;
    /* Use a temporary row buffer to avoid aliasing issues when
       tw < sw -- writing to the same buffer we're reading from. */
    for (uint32_t ty = 0; ty < th; ty++) {
        uint32_t sy0 = ty * sh / th;
        uint32_t sy1 = (ty + 1) * sh / th;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (uint32_t tx = 0; tx < tw; tx++) {
            uint32_t sx0 = tx * sw / tw;
            uint32_t sx1 = (tx + 1) * sw / tw;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            uint32_t count = 0;
            uint32_t acc[4] = {0, 0, 0, 0};
            for (uint32_t sy = sy0; sy < sy1 && sy < sh; sy++) {
                for (uint32_t sx = sx0; sx < sx1 && sx < sw; sx++) {
                    uint32_t off = (sy * sw + sx) * C;
                    for (uint32_t c = 0; c < C; c++)
                        acc[c] += buf[off + c];
                    count++;
                }
            }
            uint32_t dst_off = (ty * tw + tx) * C;
            if (count > 0) {
                for (uint32_t c = 0; c < C; c++)
                    buf[dst_off + c] = (uint8_t)(acc[c] / count);
            }
        }
    }
}

NODUS_API int nodus_scrub_ring_reduce_output(
        NodusScrubRing *ring, int32_t index,
        uint32_t target_w, uint32_t target_h)
{
    if (!ring || index < 0 || index >= ring->length) return -1;

    ring_lock(ring);
    int bi = scrub_buf_index(ring, index);
    NodusScrubEntry *e = &ring->entries[bi];

    if (target_w == 0 || target_h == 0) {
        /* Clear the output image entirely. */
        e->flags &= ~(uint32_t)NODUS_SCRUB_FLAG_HAS_OUTPUT;
        e->output_w = 0;
        e->output_h = 0;
        e->flags |= NODUS_SCRUB_FLAG_REDUCED;
        ring_unlock(ring);
        return 0;
    }

    if (target_w > NODUS_SCRUB_IMAGE_W) target_w = NODUS_SCRUB_IMAGE_W;
    if (target_h > NODUS_SCRUB_IMAGE_H) target_h = NODUS_SCRUB_IMAGE_H;

    if (e->output_w > 0 && e->output_h > 0
        && (e->flags & NODUS_SCRUB_FLAG_HAS_OUTPUT)) {
        downsample_box_rgba(e->output_image,
                            e->output_w, e->output_h,
                            target_w, target_h);
        e->output_w = target_w;
        e->output_h = target_h;
    }
    e->flags |= NODUS_SCRUB_FLAG_REDUCED;
    ring_unlock(ring);
    return 0;
}

NODUS_API int nodus_scrub_ring_reduce_training(
        NodusScrubRing *ring, int32_t index,
        uint32_t target_w, uint32_t target_h)
{
    if (!ring || index < 0 || index >= ring->length) return -1;

    ring_lock(ring);
    int bi = scrub_buf_index(ring, index);
    NodusScrubEntry *e = &ring->entries[bi];

    if (target_w == 0 || target_h == 0) {
        e->flags &= ~(uint32_t)NODUS_SCRUB_FLAG_HAS_IMAGE;
        e->image_w = 0;
        e->image_h = 0;
        e->flags |= NODUS_SCRUB_FLAG_REDUCED;
        ring_unlock(ring);
        return 0;
    }

    if (target_w > NODUS_SCRUB_IMAGE_W) target_w = NODUS_SCRUB_IMAGE_W;
    if (target_h > NODUS_SCRUB_IMAGE_H) target_h = NODUS_SCRUB_IMAGE_H;

    if (e->image_w > 0 && e->image_h > 0
        && (e->flags & NODUS_SCRUB_FLAG_HAS_IMAGE)) {
        downsample_box_rgba(e->training_image,
                            e->image_w, e->image_h,
                            target_w, target_h);
        e->image_w = target_w;
        e->image_h = target_h;
    }
    e->flags |= NODUS_SCRUB_FLAG_REDUCED;
    ring_unlock(ring);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Scrub Ring locking                                                */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_scrub_ring_lock(NodusScrubRing *ring) {
    if (ring) ring_lock(ring);
}

NODUS_API void nodus_scrub_ring_unlock(NodusScrubRing *ring) {
    if (ring) ring_unlock(ring);
}

/* ================================================================== */
/*  Composite Frame Builder + GUI-side Cache                          */
/* ================================================================== */

/* Downsample src_w x src_h RGBA (4-byte) → dst_w x dst_h RGB (3-byte).
    Box-average filter with alpha application (RGB is multiplied by A against
    black) so mask-in-alpha previews remain visible after conversion.
    src and dst must NOT alias. */
static void downsample_rgba_to_rgb(
        const uint8_t *src, uint32_t src_w, uint32_t src_h,
        uint8_t *dst, uint32_t dst_w, uint32_t dst_h)
{
    if (!src || !dst || !src_w || !src_h || !dst_w || !dst_h) return;
    for (uint32_t dy = 0; dy < dst_h; dy++) {
        uint32_t sy0 = dy * src_h / dst_h;
        uint32_t sy1 = (dy + 1) * src_h / dst_h;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (uint32_t dx = 0; dx < dst_w; dx++) {
            uint32_t sx0 = dx * src_w / dst_w;
            uint32_t sx1 = (dx + 1) * src_w / dst_w;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            uint64_t acc_r = 0, acc_g = 0, acc_b = 0;
            uint32_t count = 0;
            for (uint32_t sy = sy0; sy < sy1 && sy < src_h; sy++) {
                for (uint32_t sx = sx0; sx < sx1 && sx < src_w; sx++) {
                    uint32_t off = (sy * src_w + sx) * 4;
                    uint32_t a = (uint32_t)src[off + 3];
                    acc_r += (uint64_t)src[off + 0] * (uint64_t)a;
                    acc_g += (uint64_t)src[off + 1] * (uint64_t)a;
                    acc_b += (uint64_t)src[off + 2] * (uint64_t)a;
                    count++;
                }
            }
            uint32_t dst_off = (dy * dst_w + dx) * 3;
            if (count > 0) {
                uint64_t denom = (uint64_t)count * 255ULL;
                dst[dst_off + 0] = (uint8_t)(acc_r / denom);
                dst[dst_off + 1] = (uint8_t)(acc_g / denom);
                dst[dst_off + 2] = (uint8_t)(acc_b / denom);
            }
        }
    }
}

NODUS_API int nodus_composite_build_frame(
        const NodusScrubRing *ring, int32_t index,
        uint32_t panel_w, uint32_t panel_h,
        NodusCompositeFrame *out_frame)
{
    NodusScrubRing *r = (NodusScrubRing*)ring;
    int bi;
    const NodusScrubEntry *e;
    uint32_t src_w, src_h, out_w, out_h;

    if (!r || !out_frame) return -1;
    if (panel_w == 0 || panel_h == 0) return -1;
    if (panel_w > NODUS_COMPOSITE_PANEL_MAX_W) panel_w = NODUS_COMPOSITE_PANEL_MAX_W;
    if (panel_h > NODUS_COMPOSITE_PANEL_MAX_H) panel_h = NODUS_COMPOSITE_PANEL_MAX_H;

    ring_lock(r);
    if (index < 0 || index >= r->length) {
        ring_unlock(r);
        return -1;
    }

    bi = scrub_buf_index(r, index);
    e = &r->entries[bi];

    out_frame->source_ring_cursor = r->write_cursor;
    out_frame->step      = e->step;
    out_frame->round_id  = e->round_id;
    out_frame->ts        = e->ts;
    out_frame->loss      = e->loss;
    out_frame->panel_w   = panel_w;
    out_frame->panel_h   = panel_h;
    out_frame->flags     = e->flags;

    src_w = e->image_w;
    src_h = e->image_h;
    out_w = e->output_w;
    out_h = e->output_h;

    /* Target → target_panel (RGBA source in target_data). */
    if ((e->flags & NODUS_SCRUB_FLAG_HAS_TARGET) && src_w > 0 && src_h > 0) {
        downsample_rgba_to_rgb(e->target_data, src_w, src_h,
                               out_frame->target_panel, panel_w, panel_h);
    } else {
        memset(out_frame->target_panel, 14, panel_w * panel_h * 3);
    }

    /* Training input → input_panel. */
    if ((e->flags & NODUS_SCRUB_FLAG_HAS_IMAGE) && src_w > 0 && src_h > 0) {
        downsample_rgba_to_rgb(e->training_image, src_w, src_h,
                               out_frame->input_panel, panel_w, panel_h);
    } else {
        memset(out_frame->input_panel, 14, panel_w * panel_h * 3);
    }

    /* Network output → output_panel. */
    if ((e->flags & NODUS_SCRUB_FLAG_HAS_OUTPUT) && out_w > 0 && out_h > 0) {
        downsample_rgba_to_rgb(e->output_image, out_w, out_h,
                               out_frame->output_panel, panel_w, panel_h);
    } else {
        memset(out_frame->output_panel, 14, panel_w * panel_h * 3);
    }

    ring_unlock(r);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Composite cache (local malloc ring)                               */
/* ------------------------------------------------------------------ */

typedef struct NodusCompositeCache {
    int32_t capacity;
    int32_t length;
    int32_t head;
    int32_t write_cursor;
    NodusCompositeFrame entries[];   /* flexible array member */
} NodusCompositeCache;

static int cc_buf_index(const NodusCompositeCache *c, int32_t logical) {
    return (c->head + logical) % c->capacity;
}

NODUS_API NodusCompositeCache* nodus_composite_cache_create(int32_t capacity) {
    if (capacity <= 0) capacity = NODUS_COMPOSITE_CACHE_CAPACITY;
    size_t sz = sizeof(NodusCompositeCache)
              + (size_t)capacity * sizeof(NodusCompositeFrame);
    NodusCompositeCache *c = (NodusCompositeCache *)calloc(1, sz);
    if (!c) return NULL;
    c->capacity = capacity;
    c->length = 0;
    c->head = 0;
    c->write_cursor = 0;
    return c;
}

NODUS_API void nodus_composite_cache_destroy(NodusCompositeCache *cache) {
    free(cache);
}

NODUS_API void nodus_composite_cache_clear(NodusCompositeCache *cache) {
    if (!cache) return;
    cache->length = 0;
    cache->head = 0;
    cache->write_cursor = 0;
}

NODUS_API int32_t nodus_composite_cache_push(
        NodusCompositeCache *cache,
        const NodusCompositeFrame *frame)
{
    if (!cache || !frame) return -1;

    int write_pos;
    if (cache->length < cache->capacity) {
        write_pos = cc_buf_index(cache, cache->length);
        cache->length++;
    } else {
        write_pos = cache->head;
        cache->head = (cache->head + 1) % cache->capacity;
    }

    memcpy(&cache->entries[write_pos], frame, sizeof(NodusCompositeFrame));
    return cache->write_cursor++;
}

NODUS_API int32_t nodus_composite_build_and_push(
        const NodusScrubRing *ring, int32_t ring_index,
        uint32_t panel_w, uint32_t panel_h,
        NodusCompositeCache *cache)
{
    if (!ring || !cache) return -1;

    NodusCompositeFrame frame;
    int rc = nodus_composite_build_frame(ring, ring_index, panel_w, panel_h, &frame);
    if (rc != 0) return -1;

    return nodus_composite_cache_push(cache, &frame);
}

NODUS_API int32_t nodus_composite_cache_length(const NodusCompositeCache *cache) {
    return cache ? cache->length : 0;
}

NODUS_API int32_t nodus_composite_cache_capacity(const NodusCompositeCache *cache) {
    return cache ? cache->capacity : 0;
}

NODUS_API int nodus_composite_cache_get_meta(
        const NodusCompositeCache *cache, int32_t index,
        int32_t *out_step, int32_t *out_round_id, double *out_ts,
        float *out_loss, uint32_t *out_flags,
        uint32_t *out_panel_w, uint32_t *out_panel_h,
        int32_t *out_source_ring_cursor)
{
    if (!cache || index < 0 || index >= cache->length) return -1;

    int bi = cc_buf_index(cache, index);
    const NodusCompositeFrame *f = &cache->entries[bi];

    if (out_step)      *out_step      = f->step;
    if (out_round_id)  *out_round_id  = f->round_id;
    if (out_ts)        *out_ts        = f->ts;
    if (out_loss)      *out_loss      = f->loss;
    if (out_flags)     *out_flags     = f->flags;
    if (out_panel_w)   *out_panel_w   = f->panel_w;
    if (out_panel_h)   *out_panel_h   = f->panel_h;
    if (out_source_ring_cursor) *out_source_ring_cursor = f->source_ring_cursor;

    return 0;
}

NODUS_API int32_t nodus_composite_cache_copy_panel(
        const NodusCompositeCache *cache, int32_t index, int panel_idx,
        uint8_t *out_buf, uint32_t buf_size)
{
    if (!cache || !out_buf || index < 0 || index >= cache->length) return -1;
    if (panel_idx < 0 || panel_idx > 2) return -1;

    int bi = cc_buf_index(cache, index);
    const NodusCompositeFrame *f = &cache->entries[bi];
    uint32_t panel_bytes = f->panel_w * f->panel_h * NODUS_COMPOSITE_PANEL_C;
    if (panel_bytes > NODUS_COMPOSITE_PANEL_MAX_BYTES)
        panel_bytes = NODUS_COMPOSITE_PANEL_MAX_BYTES;

    const uint8_t *src = NULL;
    switch (panel_idx) {
        case 0: src = f->target_panel; break;
        case 1: src = f->input_panel;  break;
        case 2: src = f->output_panel; break;
        default: return -1;
    }

    uint32_t n = (panel_bytes < buf_size) ? panel_bytes : buf_size;
    memcpy(out_buf, src, n);
    return (int32_t)n;
}

/* ==================================================================== */
/*  Weight state store + rendered image cache                           */
/* ==================================================================== */

#define NODUS_WEIGHT_STATE_INIT_MAGIC    0x57535453u  /* 'WSTS' */
#define NODUS_WEIGHT_IMAGE_INIT_MAGIC    0x57494D47u  /* 'WIMG' */
#define NODUS_WEIGHT_BLOB_MAGIC          0x57534231u  /* 'WSB1' */
#define NODUS_WEIGHT_STATE_MAX_PARAMS    4096

#define NODUS_WEIGHT_STATE_SHM_NAME      "NodusWeightStateStore_v3"
#define NODUS_WEIGHT_STATE_MTX_NAME      "NodusWeightStateStoreMutex_v3"
#define NODUS_WEIGHT_IMAGE_SHM_NAME      "NodusWeightImageStore_v3"
#define NODUS_WEIGHT_IMAGE_MTX_NAME      "NodusWeightImageStoreMutex_v3"

typedef struct NodusMappedBlobHandle {
#ifdef _WIN32
    HANDLE   mapping;
#else
    int      fd;
#endif
    void    *ptr;
    uint64_t size;
    char     name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
} NodusMappedBlobHandle;

typedef struct NodusWeightStateBlobHeader {
    uint32_t magic;
    uint32_t entry_count;
    uint64_t publish_seq;
    uint64_t generation;
    uint64_t architecture_version;
    int32_t  round_id;
    int32_t  cycle;
    int32_t  step;
    char     model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char     node_id[NODUS_WEIGHT_STORE_NAME_MAX];
} NodusWeightStateBlobHeader;

typedef struct NodusWeightStateBlobEntry {
    uint64_t data_offset;
    uint64_t numel;
    uint32_t shape0;
    uint32_t name_offset;
    uint32_t name_len;
    uint32_t reserved;
} NodusWeightStateBlobEntry;

typedef struct NodusWeightImageEntry {
    uint64_t image_seq;
    uint64_t state_publish_seq;
    uint64_t generation;
    uint64_t architecture_version;
    uint64_t byte_count;
    int32_t  round_id;
    int32_t  cycle;
    int32_t  step;
    int32_t  w;
    int32_t  h;
    int32_t  c;
    int32_t  stride_bytes;
    int32_t  mode;
    int32_t  target_w;
    int32_t  target_h;
    uint32_t flags;
    char     model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char     node_id[NODUS_WEIGHT_STORE_NAME_MAX];
    char     blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
} NodusWeightImageEntry;

typedef struct NodusWeightImageActiveConfig {
    uint64_t state_publish_seq;
    uint64_t generation;
    uint64_t architecture_version;
    int32_t  round_id;
    int32_t  cycle;
    int32_t  step;
    int32_t  mode;
    int32_t  target_w;
    int32_t  target_h;
    int32_t  render_w;
    int32_t  render_h;
    int32_t  render_c;
    int32_t  render_stride_bytes;
    char     model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char     node_id[NODUS_WEIGHT_STORE_NAME_MAX];
} NodusWeightImageActiveConfig;

typedef struct NodusWeightStateEntry {
    uint64_t publish_seq;
    uint64_t generation;
    uint64_t architecture_version;
    uint64_t blob_epoch;
    uint64_t blob_bytes;
    int32_t  round_id;
    int32_t  cycle;
    int32_t  step;
    int32_t  param_count;
    char     model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char     node_id[NODUS_WEIGHT_STORE_NAME_MAX];
    char     blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
} NodusWeightStateEntry;

struct NodusWeightStateStore {
    nodus_mutex_t        lock;
    uint32_t             initialized;
    int32_t              entry_count;
    uint64_t             next_publish_seq;
    uint64_t             next_blob_epoch;
    NodusWeightStateEntry entries[NODUS_WEIGHT_STATE_REGISTRY_CAPACITY];
};

struct NodusWeightImageStore {
    nodus_mutex_t         lock;
    uint32_t              initialized;
    int32_t               max_entries;
    int32_t               entry_count;
    uint64_t              max_total_bytes;
    uint64_t              total_bytes;
    uint64_t              next_image_seq;
    NodusWeightImageActiveConfig active_config;
    NodusWeightImageEntry entries[NODUS_WEIGHT_IMAGE_CACHE_CAPACITY];
};

static NodusWeightStateStore *g_weight_state_global = NULL;
static NodusWeightImageStore *g_weight_image_global = NULL;
#ifdef _WIN32
static HANDLE g_weight_state_shm_mutex = NULL;
static HANDLE g_weight_image_shm_mutex = NULL;
#endif
static NodusMappedBlobHandle g_weight_state_entry_blobs[NODUS_WEIGHT_STATE_REGISTRY_CAPACITY] = {0};
static NodusMappedBlobHandle g_weight_image_slot_blobs[NODUS_WEIGHT_IMAGE_CACHE_CAPACITY] = {0};

static void copy_cstr_trunc(char *dst, size_t cap, const char *src) {
    size_t i = 0;
    if (!dst || cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    while (src[i] != '\0' && i + 1 < cap) {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
}

static uint64_t align_u64(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    return (v + (a - 1u)) & ~(a - 1u);
}

static int copy_out_string(const char *src, char *dst, int dst_len) {
    if (!dst || dst_len <= 0) return 0;
    copy_cstr_trunc(dst, (size_t)dst_len, src ? src : "");
    return 0;
}

static int format_blob_os_name(char *out, size_t out_cap, const char *name) {
    if (!out || out_cap == 0 || !name || !name[0]) return -1;
#ifdef _WIN32
    copy_cstr_trunc(out, out_cap, name);
#else
    if (name[0] == '/') {
        copy_cstr_trunc(out, out_cap, name);
    } else {
        if (snprintf(out, out_cap, "/%s", name) < 0) return -1;
    }
#endif
    return 0;
}

static void mapped_blob_close(NodusMappedBlobHandle *blob) {
    if (!blob) return;
#ifdef _WIN32
    if (blob->ptr) {
        UnmapViewOfFile(blob->ptr);
    }
    if (blob->mapping) {
        CloseHandle(blob->mapping);
    }
    blob->mapping = NULL;
#else
    if (blob->ptr && blob->size > 0) {
        munmap(blob->ptr, (size_t)blob->size);
    }
    if (blob->name[0] != '\0' && blob->fd >= 0) {
        close(blob->fd);
    }
    blob->fd = -1;
#endif
    blob->ptr = NULL;
    blob->size = 0;
    blob->name[0] = '\0';
}

static int mapped_blob_create_rw(NodusMappedBlobHandle *blob, const char *name, uint64_t size) {
    char os_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX + 8];
    if (!blob || !name || size == 0) return -1;
    memset(blob, 0, sizeof(*blob));
#ifndef _WIN32
    blob->fd = -1;
#endif
    if (format_blob_os_name(os_name, sizeof(os_name), name) != 0) return -1;
#ifdef _WIN32
    {
        DWORD size_lo = (DWORD)(size & 0xffffffffu);
        DWORD size_hi = (DWORD)((size >> 32u) & 0xffffffffu);
        HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, size_hi, size_lo, os_name);
        if (!mapping) return -1;
        void *ptr = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)size);
        if (!ptr) {
            CloseHandle(mapping);
            return -1;
        }
        blob->mapping = mapping;
        blob->ptr = ptr;
    }
#else
    {
        int fd = shm_open(os_name, O_CREAT | O_RDWR, 0600);
        if (fd < 0) return -1;
        if (ftruncate(fd, (off_t)size) != 0) {
            close(fd);
            return -1;
        }
        void *ptr = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            return -1;
        }
        blob->fd = fd;
        blob->ptr = ptr;
    }
#endif
    blob->size = size;
    copy_cstr_trunc(blob->name, sizeof(blob->name), name);
    return 0;
}

static int mapped_blob_open_ro(NodusMappedBlobHandle *blob, const char *name, uint64_t size_hint) {
    char os_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX + 8];
    if (!blob || !name || !name[0]) return -1;
    memset(blob, 0, sizeof(*blob));
#ifndef _WIN32
    blob->fd = -1;
#endif
    if (format_blob_os_name(os_name, sizeof(os_name), name) != 0) return -1;
#ifdef _WIN32
    {
        HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, os_name);
        if (!mapping) return -1;
        void *ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, (SIZE_T)size_hint);
        if (!ptr) {
            CloseHandle(mapping);
            return -1;
        }
        blob->mapping = mapping;
        blob->ptr = ptr;
    }
#else
    {
        int fd = shm_open(os_name, O_RDONLY, 0600);
        if (fd < 0) return -1;
        void *ptr = mmap(NULL, (size_t)size_hint, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            return -1;
        }
        blob->fd = fd;
        blob->ptr = ptr;
    }
#endif
    blob->size = size_hint;
    copy_cstr_trunc(blob->name, sizeof(blob->name), name);
    return 0;
}

static void weight_state_store_init(NodusWeightStateStore *store, int cross_process) {
#ifdef _WIN32
    (void)cross_process;
    mutex_init(&store->lock);
#else
    if (cross_process) mutex_init_shared(&store->lock);
    else mutex_init(&store->lock);
#endif
    store->initialized = NODUS_WEIGHT_STATE_INIT_MAGIC;
    store->entry_count = 0;
    store->next_publish_seq = 0u;
    store->next_blob_epoch = 0u;
    memset(store->entries, 0, sizeof(store->entries));
}

static void weight_image_store_init(NodusWeightImageStore *store, int cross_process) {
    int i;
#ifdef _WIN32
    (void)cross_process;
    mutex_init(&store->lock);
#else
    if (cross_process) mutex_init_shared(&store->lock);
    else mutex_init(&store->lock);
#endif
    store->initialized = NODUS_WEIGHT_IMAGE_INIT_MAGIC;
    store->max_entries = NODUS_WEIGHT_IMAGE_CACHE_CAPACITY;
    store->entry_count = 0;
    store->max_total_bytes = NODUS_WEIGHT_IMAGE_DEFAULT_MAX_BYTES;
    store->total_bytes = 0u;
    store->next_image_seq = 0;
    memset(&store->active_config, 0, sizeof(store->active_config));
    for (i = 0; i < NODUS_WEIGHT_IMAGE_CACHE_CAPACITY; i++) {
        memset(&store->entries[i], 0, sizeof(store->entries[i]));
    }
}

#ifdef _WIN32
static void weight_state_store_lock(NodusWeightStateStore *store) {
    if (store == g_weight_state_global && g_weight_state_shm_mutex)
        WaitForSingleObject(g_weight_state_shm_mutex, INFINITE);
    else
        EnterCriticalSection(&store->lock);
}
static void weight_state_store_unlock(NodusWeightStateStore *store) {
    if (store == g_weight_state_global && g_weight_state_shm_mutex)
        ReleaseMutex(g_weight_state_shm_mutex);
    else
        LeaveCriticalSection(&store->lock);
}
static void weight_image_store_lock(NodusWeightImageStore *store) {
    if (store == g_weight_image_global && g_weight_image_shm_mutex)
        WaitForSingleObject(g_weight_image_shm_mutex, INFINITE);
    else
        EnterCriticalSection(&store->lock);
}
static void weight_image_store_unlock(NodusWeightImageStore *store) {
    if (store == g_weight_image_global && g_weight_image_shm_mutex)
        ReleaseMutex(g_weight_image_shm_mutex);
    else
        LeaveCriticalSection(&store->lock);
}
#else
static void weight_state_store_lock(NodusWeightStateStore *store)   { pthread_mutex_lock(&store->lock); }
static void weight_state_store_unlock(NodusWeightStateStore *store) { pthread_mutex_unlock(&store->lock); }
static void weight_image_store_lock(NodusWeightImageStore *store)   { pthread_mutex_lock(&store->lock); }
static void weight_image_store_unlock(NodusWeightImageStore *store) { pthread_mutex_unlock(&store->lock); }
#endif

static void weight_image_store_zero_entry(NodusWeightImageEntry *entry) {
    if (!entry) return;
    memset(entry, 0, sizeof(*entry));
}

static void mapped_blob_zero(NodusMappedBlobHandle *blob) {
    if (!blob) return;
    memset(blob, 0, sizeof(*blob));
#ifndef _WIN32
    blob->fd = -1;
#endif
}

static void weight_state_store_zero_entry(NodusWeightStateEntry *entry) {
    if (!entry) return;
    memset(entry, 0, sizeof(*entry));
}

static int weight_state_store_find_locked(
        const NodusWeightStateStore *store,
        const char *model_name,
        const char *node_id)
{
    int i;
    if (!store) return -1;
    for (i = 0; i < store->entry_count; i++) {
        if (strcmp(store->entries[i].model_name, model_name ? model_name : "") != 0) continue;
        if (strcmp(store->entries[i].node_id, node_id ? node_id : "") != 0) continue;
        return i;
    }
    return -1;
}

static void weight_state_store_remove_locked(NodusWeightStateStore *store, int logical_index) {
    int move_count;
    if (!store || logical_index < 0 || logical_index >= store->entry_count) return;
    mapped_blob_close(&g_weight_state_entry_blobs[logical_index]);
    move_count = store->entry_count - logical_index - 1;
    if (move_count > 0) {
        memmove(&store->entries[logical_index],
                &store->entries[logical_index + 1],
                (size_t)move_count * sizeof(store->entries[0]));
        memmove(&g_weight_state_entry_blobs[logical_index],
                &g_weight_state_entry_blobs[logical_index + 1],
                (size_t)move_count * sizeof(g_weight_state_entry_blobs[0]));
    }
    weight_state_store_zero_entry(&store->entries[store->entry_count - 1]);
    mapped_blob_zero(&g_weight_state_entry_blobs[store->entry_count - 1]);
    store->entry_count--;
}

static int weight_state_store_copy_entry_locked(
        const NodusWeightStateStore *store,
        int index,
        uint64_t *out_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_param_count,
        uint64_t *out_blob_bytes,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    const NodusWeightStateEntry *entry;
    if (!store || index < 0 || index >= store->entry_count) return -1;
    entry = &store->entries[index];
    if (entry->publish_seq == 0u || entry->blob_name[0] == '\0') return -1;
    if (out_publish_seq) *out_publish_seq = entry->publish_seq;
    if (out_generation) *out_generation = entry->generation;
    if (out_architecture_version) *out_architecture_version = entry->architecture_version;
    if (out_round_id) *out_round_id = entry->round_id;
    if (out_cycle) *out_cycle = entry->cycle;
    if (out_step) *out_step = entry->step;
    if (out_param_count) *out_param_count = entry->param_count;
    if (out_blob_bytes) *out_blob_bytes = entry->blob_bytes;
    copy_out_string(entry->model_name, out_model_name, out_model_name_buflen);
    copy_out_string(entry->node_id, out_node_id, out_node_id_buflen);
    copy_out_string(entry->blob_name, out_blob_name, out_blob_name_buflen);
    return 0;
}

static void weight_image_store_remove_locked(NodusWeightImageStore *store, int logical_index) {
    int move_count;
    if (!store || logical_index < 0 || logical_index >= store->entry_count) return;
    if (store->entries[logical_index].byte_count <= store->total_bytes) {
        store->total_bytes -= store->entries[logical_index].byte_count;
    } else {
        store->total_bytes = 0u;
    }
    mapped_blob_close(&g_weight_image_slot_blobs[logical_index]);
    move_count = store->entry_count - logical_index - 1;
    if (move_count > 0) {
        memmove(&store->entries[logical_index],
                &store->entries[logical_index + 1],
                (size_t)move_count * sizeof(store->entries[0]));
        memmove(&g_weight_image_slot_blobs[logical_index],
                &g_weight_image_slot_blobs[logical_index + 1],
                (size_t)move_count * sizeof(g_weight_image_slot_blobs[0]));
    }
    weight_image_store_zero_entry(&store->entries[store->entry_count - 1]);
    mapped_blob_zero(&g_weight_image_slot_blobs[store->entry_count - 1]);
    store->entry_count--;
}

static void weight_image_store_clear_locked(NodusWeightImageStore *store) {
    if (!store) return;
    while (store->entry_count > 0) {
        weight_image_store_remove_locked(store, 0);
    }
}

static int weight_image_store_choose_evict_locked(const NodusWeightImageStore *store) {
    int i;
    if (!store || store->entry_count <= 0) return -1;
    for (i = 0; i < store->entry_count; i++) {
        if ((store->entries[i].flags & NODUS_WEIGHT_IMAGE_FLAG_CHECKPOINT) == 0u) {
            return i;
        }
    }
    return 0;
}

static void weight_image_store_trim_locked(NodusWeightImageStore *store) {
    if (!store) return;
    while (store->entry_count > store->max_entries
            || (store->entry_count > 1
                && store->max_total_bytes > 0u
                && store->total_bytes > store->max_total_bytes)) {
        int evict_index = weight_image_store_choose_evict_locked(store);
        if (evict_index < 0) break;
        weight_image_store_remove_locked(store, evict_index);
    }
}

NODUS_API NodusWeightStateStore* nodus_weight_state_store_get_global(void) {
    if (g_weight_state_global) return g_weight_state_global;
#ifdef _WIN32
    {
        HANDLE shm = NULL;
        int created = 0;
        g_weight_state_shm_mutex = CreateMutexA(NULL, FALSE, NODUS_WEIGHT_STATE_MTX_NAME);
        if (!g_weight_state_shm_mutex) {
            fprintf(stderr, "[nodus] FATAL: CreateMutexA weight state failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        shm = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
            (DWORD)sizeof(NodusWeightStateStore),
            NODUS_WEIGHT_STATE_SHM_NAME
        );
        if (!shm) {
            fprintf(stderr, "[nodus] FATAL: CreateFileMappingA weight state failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        created = (GetLastError() != ERROR_ALREADY_EXISTS);
        g_weight_state_global = (NodusWeightStateStore*)MapViewOfFile(shm, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NodusWeightStateStore));
        if (!g_weight_state_global) {
            fprintf(stderr, "[nodus] FATAL: MapViewOfFile weight state failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        if (created || g_weight_state_global->initialized != NODUS_WEIGHT_STATE_INIT_MAGIC) {
            memset(g_weight_state_global, 0, sizeof(NodusWeightStateStore));
            weight_state_store_init(g_weight_state_global, 1);
        }
    }
#else
    {
        int fd = shm_open("/" NODUS_WEIGHT_STATE_SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            perror("[nodus] FATAL: shm_open weight state");
            abort();
        }
        if (ftruncate(fd, (off_t)sizeof(NodusWeightStateStore)) != 0) {
            perror("[nodus] FATAL: ftruncate weight state");
            abort();
        }
        g_weight_state_global = (NodusWeightStateStore*)mmap(NULL, sizeof(NodusWeightStateStore), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (g_weight_state_global == MAP_FAILED) {
            perror("[nodus] FATAL: mmap weight state");
            abort();
        }
        close(fd);
        if (g_weight_state_global->initialized != NODUS_WEIGHT_STATE_INIT_MAGIC) {
            memset(g_weight_state_global, 0, sizeof(NodusWeightStateStore));
            weight_state_store_init(g_weight_state_global, 1);
        }
    }
#endif
    return g_weight_state_global;
}

NODUS_API NodusWeightImageStore* nodus_weight_image_store_get_global(void) {
    if (g_weight_image_global) return g_weight_image_global;
#ifdef _WIN32
    {
        HANDLE shm = NULL;
        int created = 0;
        g_weight_image_shm_mutex = CreateMutexA(NULL, FALSE, NODUS_WEIGHT_IMAGE_MTX_NAME);
        if (!g_weight_image_shm_mutex) {
            fprintf(stderr, "[nodus] FATAL: CreateMutexA weight image failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        shm = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
            (DWORD)sizeof(NodusWeightImageStore),
            NODUS_WEIGHT_IMAGE_SHM_NAME
        );
        if (!shm) {
            fprintf(stderr, "[nodus] FATAL: CreateFileMappingA weight image failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        created = (GetLastError() != ERROR_ALREADY_EXISTS);
        g_weight_image_global = (NodusWeightImageStore*)MapViewOfFile(shm, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NodusWeightImageStore));
        if (!g_weight_image_global) {
            fprintf(stderr, "[nodus] FATAL: MapViewOfFile weight image failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        if (created || g_weight_image_global->initialized != NODUS_WEIGHT_IMAGE_INIT_MAGIC) {
            memset(g_weight_image_global, 0, sizeof(NodusWeightImageStore));
            weight_image_store_init(g_weight_image_global, 1);
        }
    }
#else
    {
        int fd = shm_open("/" NODUS_WEIGHT_IMAGE_SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            perror("[nodus] FATAL: shm_open weight image");
            abort();
        }
        if (ftruncate(fd, (off_t)sizeof(NodusWeightImageStore)) != 0) {
            perror("[nodus] FATAL: ftruncate weight image");
            abort();
        }
        g_weight_image_global = (NodusWeightImageStore*)mmap(NULL, sizeof(NodusWeightImageStore), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (g_weight_image_global == MAP_FAILED) {
            perror("[nodus] FATAL: mmap weight image");
            abort();
        }
        close(fd);
        if (g_weight_image_global->initialized != NODUS_WEIGHT_IMAGE_INIT_MAGIC) {
            memset(g_weight_image_global, 0, sizeof(NodusWeightImageStore));
            weight_image_store_init(g_weight_image_global, 1);
        }
    }
#endif
    return g_weight_image_global;
}

NODUS_API int nodus_weight_state_store_publish_flat(
        NodusWeightStateStore *store,
        const char *model_name,
        const char *node_id,
        int32_t round_id,
        int32_t cycle,
        int32_t step,
        uint64_t generation,
        uint64_t architecture_version,
        uint64_t publish_seq,
        const char *const *param_names,
        const float *const *param_data,
        const int32_t *param_numel,
        const int32_t *param_shape0,
        int32_t num_params)
{
    uint64_t total_size;
    uint64_t data_off;
    uint64_t blob_epoch;
    uint64_t next_publish_seq;
    int32_t i;
    int existing_index;
    NodusMappedBlobHandle blob = {0};
    NodusWeightStateBlobHeader *hdr = NULL;
    NodusWeightStateBlobEntry *entries = NULL;
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    if (!store || !param_names || !param_data || !param_numel || !param_shape0) return -1;
    if (num_params <= 0 || num_params > NODUS_WEIGHT_STATE_MAX_PARAMS) return -1;

    total_size = sizeof(NodusWeightStateBlobHeader)
               + ((uint64_t)num_params * (uint64_t)sizeof(NodusWeightStateBlobEntry));
    total_size = align_u64(total_size, 8u);
    for (i = 0; i < num_params; i++) {
        size_t name_len;
        if (!param_names[i] || !param_data[i] || param_numel[i] <= 0 || param_shape0[i] <= 0) return -1;
        name_len = strlen(param_names[i]);
        total_size += (uint64_t)(name_len + 1u);
        total_size = align_u64(total_size, 8u);
        total_size += ((uint64_t)param_numel[i] * (uint64_t)sizeof(float));
        total_size = align_u64(total_size, 8u);
    }

    weight_state_store_lock(store);
    blob_epoch = store->next_blob_epoch + 1u;
    next_publish_seq = store->next_publish_seq + 1u;
    if (publish_seq == 0u) publish_seq = next_publish_seq;
    if (snprintf(blob_name, sizeof(blob_name), "NodusWeightStateBlob_%llu", (unsigned long long)blob_epoch) < 0) {
        weight_state_store_unlock(store);
        return -1;
    }
    if (mapped_blob_create_rw(&blob, blob_name, total_size) != 0) {
        weight_state_store_unlock(store);
        return -1;
    }

    memset(blob.ptr, 0, (size_t)total_size);
    hdr = (NodusWeightStateBlobHeader*)blob.ptr;
    entries = (NodusWeightStateBlobEntry*)((uint8_t*)blob.ptr + sizeof(NodusWeightStateBlobHeader));
    hdr->magic = NODUS_WEIGHT_BLOB_MAGIC;
    hdr->entry_count = (uint32_t)num_params;
    hdr->publish_seq = publish_seq;
    hdr->generation = generation;
    hdr->architecture_version = architecture_version;
    hdr->round_id = round_id;
    hdr->cycle = cycle;
    hdr->step = step;
    copy_cstr_trunc(hdr->model_name, sizeof(hdr->model_name), model_name ? model_name : "");
    copy_cstr_trunc(hdr->node_id, sizeof(hdr->node_id), node_id ? node_id : "");

    data_off = align_u64(
        sizeof(NodusWeightStateBlobHeader) + ((uint64_t)num_params * (uint64_t)sizeof(NodusWeightStateBlobEntry)),
        8u
    );
    for (i = 0; i < num_params; i++) {
        uint64_t name_len = (uint64_t)strlen(param_names[i]);
        entries[i].name_offset = (uint32_t)data_off;
        entries[i].name_len = (uint32_t)name_len;
        memcpy((uint8_t*)blob.ptr + data_off, param_names[i], (size_t)name_len + 1u);
        data_off += name_len + 1u;
        data_off = align_u64(data_off, 8u);
        entries[i].data_offset = data_off;
        entries[i].numel = (uint64_t)param_numel[i];
        entries[i].shape0 = (uint32_t)param_shape0[i];
        memcpy((uint8_t*)blob.ptr + data_off, param_data[i], (size_t)param_numel[i] * sizeof(float));
        data_off += ((uint64_t)param_numel[i] * (uint64_t)sizeof(float));
        data_off = align_u64(data_off, 8u);
    }

    existing_index = weight_state_store_find_locked(store, model_name, node_id);
    if (existing_index >= 0) {
        weight_state_store_remove_locked(store, existing_index);
    } else if (store->entry_count >= NODUS_WEIGHT_STATE_REGISTRY_CAPACITY) {
        weight_state_store_remove_locked(store, 0);
    }
    if (store->entry_count >= NODUS_WEIGHT_STATE_REGISTRY_CAPACITY) {
        mapped_blob_close(&blob);
        weight_state_store_unlock(store);
        return -1;
    }
    g_weight_state_entry_blobs[store->entry_count] = blob;
    weight_state_store_zero_entry(&store->entries[store->entry_count]);
    store->entries[store->entry_count].publish_seq = publish_seq;
    store->entries[store->entry_count].generation = generation;
    store->entries[store->entry_count].architecture_version = architecture_version;
    store->entries[store->entry_count].blob_epoch = blob_epoch;
    store->entries[store->entry_count].blob_bytes = total_size;
    store->entries[store->entry_count].round_id = round_id;
    store->entries[store->entry_count].cycle = cycle;
    store->entries[store->entry_count].step = step;
    store->entries[store->entry_count].param_count = num_params;
    copy_cstr_trunc(store->entries[store->entry_count].model_name, sizeof(store->entries[store->entry_count].model_name), model_name ? model_name : "");
    copy_cstr_trunc(store->entries[store->entry_count].node_id, sizeof(store->entries[store->entry_count].node_id), node_id ? node_id : "");
    copy_cstr_trunc(store->entries[store->entry_count].blob_name, sizeof(store->entries[store->entry_count].blob_name), blob_name);
    store->entry_count++;
    store->next_blob_epoch = blob_epoch;
    if (publish_seq > store->next_publish_seq) {
        store->next_publish_seq = publish_seq;
    } else {
        store->next_publish_seq = next_publish_seq;
    }
    weight_state_store_unlock(store);
    return 0;
}

NODUS_API int nodus_weight_state_store_get_meta(
        const NodusWeightStateStore *store,
        uint64_t *out_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_param_count,
        uint64_t *out_blob_bytes,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    NodusWeightStateStore *st = (NodusWeightStateStore*)store;
    if (!st) return -1;
    weight_state_store_lock(st);
    if (st->entry_count <= 0) {
        weight_state_store_unlock(st);
        return -1;
    }
    if (weight_state_store_copy_entry_locked(
            st,
            st->entry_count - 1,
            out_publish_seq,
            out_generation,
            out_architecture_version,
            out_round_id,
            out_cycle,
            out_step,
            out_param_count,
            out_blob_bytes,
            out_model_name,
            out_model_name_buflen,
            out_node_id,
            out_node_id_buflen,
            out_blob_name,
            out_blob_name_buflen
        ) != 0) {
        weight_state_store_unlock(st);
        return -1;
    }
    weight_state_store_unlock(st);
    return 0;
}

NODUS_API int32_t nodus_weight_state_store_count(const NodusWeightStateStore *store) {
    NodusWeightStateStore *st = (NodusWeightStateStore*)store;
    int32_t count = 0;
    if (!st) return 0;
    weight_state_store_lock(st);
    count = st->entry_count;
    weight_state_store_unlock(st);
    return count;
}

NODUS_API int nodus_weight_state_store_get_meta_at(
        const NodusWeightStateStore *store,
        int32_t index,
        uint64_t *out_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_param_count,
        uint64_t *out_blob_bytes,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    NodusWeightStateStore *st = (NodusWeightStateStore*)store;
    int rc;
    if (!st) return -1;
    weight_state_store_lock(st);
    rc = weight_state_store_copy_entry_locked(
        st,
        (int)index,
        out_publish_seq,
        out_generation,
        out_architecture_version,
        out_round_id,
        out_cycle,
        out_step,
        out_param_count,
        out_blob_bytes,
        out_model_name,
        out_model_name_buflen,
        out_node_id,
        out_node_id_buflen,
        out_blob_name,
        out_blob_name_buflen
    );
    weight_state_store_unlock(st);
    return rc;
}

NODUS_API int nodus_weight_state_store_get_meta_for(
        const NodusWeightStateStore *store,
        const char *model_name,
        const char *node_id,
        uint64_t *out_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_param_count,
        uint64_t *out_blob_bytes,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    NodusWeightStateStore *st = (NodusWeightStateStore*)store;
    int index;
    int rc;
    if (!st) return -1;
    weight_state_store_lock(st);
    index = weight_state_store_find_locked(st, model_name, node_id);
    if (index < 0) {
        weight_state_store_unlock(st);
        return -1;
    }
    rc = weight_state_store_copy_entry_locked(
        st,
        index,
        out_publish_seq,
        out_generation,
        out_architecture_version,
        out_round_id,
        out_cycle,
        out_step,
        out_param_count,
        out_blob_bytes,
        NULL,
        0,
        NULL,
        0,
        out_blob_name,
        out_blob_name_buflen
    );
    weight_state_store_unlock(st);
    return rc;
}

static int weight_state_store_select_meta(
        const NodusWeightStateStore *state_store,
        const char *model_name,
        const char *node_id,
        uint64_t *out_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_param_count,
        uint64_t *out_blob_bytes,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    if (model_name && model_name[0] != '\0') {
        int rc = nodus_weight_state_store_get_meta_for(
            state_store,
            model_name,
            node_id,
            out_publish_seq,
            out_generation,
            out_architecture_version,
            out_round_id,
            out_cycle,
            out_step,
            out_param_count,
            out_blob_bytes,
            out_blob_name,
            out_blob_name_buflen
        );
        if (rc != 0) return rc;
        copy_out_string(model_name, out_model_name, out_model_name_buflen);
        copy_out_string(node_id ? node_id : "", out_node_id, out_node_id_buflen);
        return 0;
    }
    return nodus_weight_state_store_get_meta(
        state_store,
        out_publish_seq,
        out_generation,
        out_architecture_version,
        out_round_id,
        out_cycle,
        out_step,
        out_param_count,
        out_blob_bytes,
        out_model_name,
        out_model_name_buflen,
        out_node_id,
        out_node_id_buflen,
        out_blob_name,
        out_blob_name_buflen
    );
}

static int weight_image_store_publish_rgb(
        NodusWeightImageStore *store,
        const char *model_name,
        const char *node_id,
        int32_t round_id,
        int32_t cycle,
        int32_t step,
        uint64_t state_publish_seq,
        uint64_t generation,
        uint64_t architecture_version,
        const uint8_t *rgb,
        int32_t w,
        int32_t h,
        int32_t c,
        int32_t mode,
        int32_t target_w,
        int32_t target_h,
        uint32_t flags)
{
    uint64_t byte_count;
    uint64_t image_seq;
    int write_pos;
    int logical;
    uint32_t carry_flags = 0u;
    int32_t carry_round_id = round_id;
    int32_t carry_cycle = cycle;
    NodusMappedBlobHandle blob = {0};
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    NodusWeightImageEntry *entry = NULL;
    if (!store || !rgb || w <= 0 || h <= 0 || c <= 0) return -1;
    byte_count = (uint64_t)w * (uint64_t)h * (uint64_t)c;
    weight_image_store_lock(store);
    image_seq = store->next_image_seq + 1u;
    if (snprintf(blob_name, sizeof(blob_name), "NodusWeightImageBlob_%llu", (unsigned long long)image_seq) < 0) {
        weight_image_store_unlock(store);
        return -1;
    }
    if (mapped_blob_create_rw(&blob, blob_name, byte_count) != 0) {
        weight_image_store_unlock(store);
        return -1;
    }
    memcpy(blob.ptr, rgb, (size_t)byte_count);

    for (logical = store->entry_count - 1; logical >= 0; logical--) {
        if (store->entries[logical].state_publish_seq == state_publish_seq) {
            carry_flags |= store->entries[logical].flags;
            if ((store->entries[logical].flags & NODUS_WEIGHT_IMAGE_FLAG_CHECKPOINT) != 0u) {
                carry_round_id = store->entries[logical].round_id;
                carry_cycle = store->entries[logical].cycle;
            }
            weight_image_store_remove_locked(store, logical);
        }
    }
    while ((store->entry_count >= store->max_entries && store->entry_count > 0)
            || (store->max_total_bytes > 0u && (store->total_bytes + byte_count) > store->max_total_bytes
                && store->entry_count > 0)) {
        int evict_index = weight_image_store_choose_evict_locked(store);
        if (evict_index < 0) break;
        weight_image_store_remove_locked(store, evict_index);
    }
    if (store->entry_count >= NODUS_WEIGHT_IMAGE_CACHE_CAPACITY) {
        mapped_blob_close(&blob);
        weight_image_store_unlock(store);
        return -1;
    }

    write_pos = store->entry_count;
    g_weight_image_slot_blobs[write_pos] = blob;
    entry = &store->entries[write_pos];
    weight_image_store_zero_entry(entry);
    entry->image_seq = image_seq;
    entry->state_publish_seq = state_publish_seq;
    entry->generation = generation;
    entry->architecture_version = architecture_version;
    entry->byte_count = byte_count;
    entry->round_id = carry_round_id;
    entry->cycle = carry_cycle;
    entry->step = step;
    entry->w = w;
    entry->h = h;
    entry->c = c;
    entry->stride_bytes = w * c;
    entry->mode = mode;
    entry->target_w = target_w;
    entry->target_h = target_h;
    entry->flags = flags | carry_flags;
    copy_cstr_trunc(entry->model_name, sizeof(entry->model_name), model_name ? model_name : "");
    copy_cstr_trunc(entry->node_id, sizeof(entry->node_id), node_id ? node_id : "");
    copy_cstr_trunc(entry->blob_name, sizeof(entry->blob_name), blob_name);
    store->entry_count++;
    store->total_bytes += byte_count;
    store->next_image_seq = image_seq;
    weight_image_store_trim_locked(store);
    weight_image_store_unlock(store);
    return 0;
}

NODUS_API int nodus_weight_image_store_render_latest(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        int mode,
        int32_t target_w,
        int32_t target_h)
{
    return nodus_weight_image_store_render_for(
        state_store,
        image_store,
        NULL,
        NULL,
        mode,
        target_w,
        target_h
    );
}

NODUS_API int nodus_weight_image_store_measure_for(
        const NodusWeightStateStore *state_store,
        const char *model_name,
        const char *node_id,
        int mode,
        int32_t target_w,
        int32_t target_h,
        uint64_t *out_state_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_render_w,
        int32_t *out_render_h,
        int32_t *out_render_c,
        int32_t *out_render_stride_bytes)
{
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    char selected_model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char selected_node_id[NODUS_WEIGHT_STORE_NAME_MAX];
    uint64_t publish_seq = 0;
    uint64_t generation = 0;
    uint64_t architecture_version = 0;
    uint64_t blob_bytes = 0;
    int32_t round_id = 0, cycle = 0, step = 0, param_count = 0;
    NodusMappedBlobHandle blob = {0};
    const NodusWeightStateBlobHeader *hdr = NULL;
    const NodusWeightStateBlobEntry *entries = NULL;
    const char **names = NULL;
    const float **data_ptrs = NULL;
    int32_t *numel = NULL;
    int32_t *shape0 = NULL;
    int32_t out_w = 0, out_h = 0;
    int rc = -1;
    int32_t i;

    if (!state_store) return -1;
    if (weight_state_store_select_meta(
            state_store,
            model_name,
            node_id,
            &publish_seq,
            &generation,
            &architecture_version,
            &round_id,
            &cycle,
            &step,
            &param_count,
            &blob_bytes,
            selected_model_name,
            (int)sizeof(selected_model_name),
            selected_node_id,
            (int)sizeof(selected_node_id),
            blob_name,
            (int)sizeof(blob_name)
        ) != 0) {
        return -1;
    }
    if (mapped_blob_open_ro(&blob, blob_name, blob_bytes) != 0) return -1;
    hdr = (const NodusWeightStateBlobHeader*)blob.ptr;
    if (!hdr || hdr->magic != NODUS_WEIGHT_BLOB_MAGIC || hdr->entry_count == 0u) {
        mapped_blob_close(&blob);
        return -1;
    }
    entries = (const NodusWeightStateBlobEntry*)((const uint8_t*)blob.ptr + sizeof(NodusWeightStateBlobHeader));
    names = (const char**)calloc((size_t)hdr->entry_count, sizeof(const char*));
    data_ptrs = (const float**)calloc((size_t)hdr->entry_count, sizeof(const float*));
    numel = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    shape0 = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    if (!names || !data_ptrs || !numel || !shape0) goto cleanup;
    for (i = 0; i < (int32_t)hdr->entry_count; i++) {
        const NodusWeightStateBlobEntry *e = &entries[i];
        if (e->name_offset >= blob.size || e->data_offset >= blob.size) goto cleanup;
        if ((e->name_offset + (uint64_t)e->name_len + 1u) > blob.size) goto cleanup;
        if ((e->data_offset + (e->numel * (uint64_t)sizeof(float))) > blob.size) goto cleanup;
        names[i] = (const char*)((const uint8_t*)blob.ptr + e->name_offset);
        data_ptrs[i] = (const float*)((const uint8_t*)blob.ptr + e->data_offset);
        numel[i] = (int32_t)e->numel;
        shape0[i] = (int32_t)e->shape0;
    }
    if (nodus_weight_image_measure_from_state_dict(
            names,
            data_ptrs,
            numel,
            shape0,
            (int32_t)hdr->entry_count,
            mode,
            target_w,
            target_h,
            &out_w,
            &out_h
        ) != 0) {
        goto cleanup;
    }
    if (out_state_publish_seq) *out_state_publish_seq = hdr->publish_seq;
    if (out_generation) *out_generation = hdr->generation;
    if (out_architecture_version) *out_architecture_version = hdr->architecture_version;
    if (out_round_id) *out_round_id = hdr->round_id;
    if (out_cycle) *out_cycle = hdr->cycle;
    if (out_step) *out_step = hdr->step;
    if (out_render_w) *out_render_w = out_w;
    if (out_render_h) *out_render_h = out_h;
    if (out_render_c) *out_render_c = 3;
    if (out_render_stride_bytes) *out_render_stride_bytes = out_w * 3;
    rc = 0;
cleanup:
    if (shape0) free(shape0);
    if (numel) free(numel);
    if (data_ptrs) free(data_ptrs);
    if (names) free(names);
    mapped_blob_close(&blob);
    return rc;
}

NODUS_API int nodus_weight_image_store_render_for(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        const char *model_name,
        const char *node_id,
        int mode,
        int32_t target_w,
        int32_t target_h)
{
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    char selected_model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char selected_node_id[NODUS_WEIGHT_STORE_NAME_MAX];
    uint64_t publish_seq = 0;
    uint64_t generation = 0;
    uint64_t architecture_version = 0;
    uint64_t blob_bytes = 0;
    int32_t round_id = 0, cycle = 0, step = 0, param_count = 0;
    NodusMappedBlobHandle blob = {0};
    const NodusWeightStateBlobHeader *hdr = NULL;
    const NodusWeightStateBlobEntry *entries = NULL;
    const char **names = NULL;
    const float **data_ptrs = NULL;
    int32_t *numel = NULL;
    int32_t *shape0 = NULL;
    uint8_t *out_rgb = NULL;
    int32_t out_w = 0, out_h = 0;
    int rc = -1;
    int32_t i;

    if (!state_store || !image_store) return -1;
    if (weight_state_store_select_meta(
            state_store,
            model_name,
            node_id,
            &publish_seq,
            &generation,
            &architecture_version,
            &round_id,
            &cycle,
            &step,
            &param_count,
            &blob_bytes,
            selected_model_name,
            (int)sizeof(selected_model_name),
            selected_node_id,
            (int)sizeof(selected_node_id),
            blob_name,
            (int)sizeof(blob_name)
        ) != 0) {
        return -1;
    }
    if (mapped_blob_open_ro(&blob, blob_name, blob_bytes) != 0) return -1;
    hdr = (const NodusWeightStateBlobHeader*)blob.ptr;
    if (!hdr || hdr->magic != NODUS_WEIGHT_BLOB_MAGIC || hdr->entry_count == 0u) {
        mapped_blob_close(&blob);
        return -1;
    }
    entries = (const NodusWeightStateBlobEntry*)((const uint8_t*)blob.ptr + sizeof(NodusWeightStateBlobHeader));
    names = (const char**)calloc((size_t)hdr->entry_count, sizeof(const char*));
    data_ptrs = (const float**)calloc((size_t)hdr->entry_count, sizeof(const float*));
    numel = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    shape0 = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    if (!names || !data_ptrs || !numel || !shape0) goto cleanup;
    for (i = 0; i < (int32_t)hdr->entry_count; i++) {
        const NodusWeightStateBlobEntry *e = &entries[i];
        if (e->name_offset >= blob.size || e->data_offset >= blob.size) goto cleanup;
        if ((e->name_offset + (uint64_t)e->name_len + 1u) > blob.size) goto cleanup;
        if ((e->data_offset + (e->numel * (uint64_t)sizeof(float))) > blob.size) goto cleanup;
        names[i] = (const char*)((const uint8_t*)blob.ptr + e->name_offset);
        data_ptrs[i] = (const float*)((const uint8_t*)blob.ptr + e->data_offset);
        numel[i] = (int32_t)e->numel;
        shape0[i] = (int32_t)e->shape0;
    }
    if (nodus_weight_image_from_state_dict(
            names,
            data_ptrs,
            numel,
            shape0,
            (int32_t)hdr->entry_count,
            NULL,
            NULL,
            mode,
            target_w,
            target_h,
            &out_rgb,
            &out_w,
            &out_h
        ) != 0) {
        goto cleanup;
    }
    rc = weight_image_store_publish_rgb(
        image_store,
        hdr->model_name,
        hdr->node_id,
        hdr->round_id,
        hdr->cycle,
        hdr->step,
        hdr->publish_seq,
        hdr->generation,
        hdr->architecture_version,
        out_rgb,
        out_w,
        out_h,
        3,
        (int32_t)mode,
        target_w,
        target_h,
        0u
    );
cleanup:
    if (out_rgb) nodus_weight_image_free(out_rgb);
    if (shape0) free(shape0);
    if (numel) free(numel);
    if (data_ptrs) free(data_ptrs);
    if (names) free(names);
    mapped_blob_close(&blob);
    return rc;
}

NODUS_API int32_t nodus_weight_image_store_length(const NodusWeightImageStore *store) {
    NodusWeightImageStore *st = (NodusWeightImageStore*)store;
    int32_t count;
    if (!st) return 0;
    weight_image_store_lock(st);
    count = st->entry_count;
    weight_image_store_unlock(st);
    return count;
}

NODUS_API int32_t nodus_weight_image_store_capacity(const NodusWeightImageStore *store) {
    NodusWeightImageStore *st = (NodusWeightImageStore*)store;
    int32_t cap;
    if (!st) return 0;
    weight_image_store_lock(st);
    cap = st->max_entries;
    weight_image_store_unlock(st);
    return cap;
}

NODUS_API int nodus_weight_image_store_set_limits(
        NodusWeightImageStore *store,
        int32_t max_entries,
        uint64_t max_total_bytes)
{
    if (!store) return -1;
    if (max_entries <= 0) max_entries = NODUS_WEIGHT_IMAGE_CACHE_CAPACITY;
    if (max_entries > NODUS_WEIGHT_IMAGE_CACHE_CAPACITY) {
        max_entries = NODUS_WEIGHT_IMAGE_CACHE_CAPACITY;
    }
    if (max_total_bytes == 0u) {
        max_total_bytes = NODUS_WEIGHT_IMAGE_DEFAULT_MAX_BYTES;
    }
    weight_image_store_lock(store);
    store->max_entries = max_entries;
    store->max_total_bytes = max_total_bytes;
    weight_image_store_trim_locked(store);
    weight_image_store_unlock(store);
    return 0;
}

NODUS_API int nodus_weight_image_store_get_stats(
        const NodusWeightImageStore *store,
        int32_t *out_max_entries,
        int32_t *out_entry_count,
        uint64_t *out_max_total_bytes,
        uint64_t *out_total_bytes)
{
    NodusWeightImageStore *st = (NodusWeightImageStore*)store;
    if (!st) return -1;
    weight_image_store_lock(st);
    if (out_max_entries) *out_max_entries = st->max_entries;
    if (out_entry_count) *out_entry_count = st->entry_count;
    if (out_max_total_bytes) *out_max_total_bytes = st->max_total_bytes;
    if (out_total_bytes) *out_total_bytes = st->total_bytes;
    weight_image_store_unlock(st);
    return 0;
}

NODUS_API int nodus_weight_image_store_configure_latest(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        int mode,
        int32_t target_w,
        int32_t target_h)
{
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    char model_name[NODUS_WEIGHT_STORE_NAME_MAX];
    char node_id[NODUS_WEIGHT_STORE_NAME_MAX];
    uint64_t publish_seq = 0;
    uint64_t generation = 0;
    uint64_t architecture_version = 0;
    uint64_t blob_bytes = 0;
    int32_t round_id = 0, cycle = 0, step = 0, param_count = 0;
    NodusMappedBlobHandle blob = {0};
    const NodusWeightStateBlobHeader *hdr = NULL;
    const NodusWeightStateBlobEntry *entries = NULL;
    const char **names = NULL;
    const float **data_ptrs = NULL;
    int32_t *numel = NULL;
    int32_t *shape0 = NULL;
    int32_t render_w = 0, render_h = 0;
    int rc = -1;
    int32_t i;

    if (!state_store || !image_store) return -1;
    if (nodus_weight_state_store_get_meta(
            state_store,
            &publish_seq,
            &generation,
            &architecture_version,
            &round_id,
            &cycle,
            &step,
            &param_count,
            &blob_bytes,
            model_name,
            (int)sizeof(model_name),
            node_id,
            (int)sizeof(node_id),
            blob_name,
            (int)sizeof(blob_name)
        ) != 0) {
        return -1;
    }
    if (mapped_blob_open_ro(&blob, blob_name, blob_bytes) != 0) return -1;
    hdr = (const NodusWeightStateBlobHeader*)blob.ptr;
    if (!hdr || hdr->magic != NODUS_WEIGHT_BLOB_MAGIC || hdr->entry_count == 0u) {
        mapped_blob_close(&blob);
        return -1;
    }
    entries = (const NodusWeightStateBlobEntry*)((const uint8_t*)blob.ptr + sizeof(NodusWeightStateBlobHeader));
    names = (const char**)calloc((size_t)hdr->entry_count, sizeof(const char*));
    data_ptrs = (const float**)calloc((size_t)hdr->entry_count, sizeof(const float*));
    numel = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    shape0 = (int32_t*)calloc((size_t)hdr->entry_count, sizeof(int32_t));
    if (!names || !data_ptrs || !numel || !shape0) goto cleanup;
    for (i = 0; i < (int32_t)hdr->entry_count; i++) {
        const NodusWeightStateBlobEntry *e = &entries[i];
        if (e->name_offset >= blob.size || e->data_offset >= blob.size) goto cleanup;
        if ((e->name_offset + (uint64_t)e->name_len + 1u) > blob.size) goto cleanup;
        if ((e->data_offset + (e->numel * (uint64_t)sizeof(float))) > blob.size) goto cleanup;
        names[i] = (const char*)((const uint8_t*)blob.ptr + e->name_offset);
        data_ptrs[i] = (const float*)((const uint8_t*)blob.ptr + e->data_offset);
        numel[i] = (int32_t)e->numel;
        shape0[i] = (int32_t)e->shape0;
    }
    if (nodus_weight_image_measure_from_state_dict(
            names,
            data_ptrs,
            numel,
            shape0,
            (int32_t)hdr->entry_count,
            mode,
            target_w,
            target_h,
            &render_w,
            &render_h
        ) != 0) {
        goto cleanup;
    }

    weight_image_store_lock(image_store);
    if (image_store->active_config.mode != (int32_t)mode
            || image_store->active_config.target_w != target_w
            || image_store->active_config.target_h != target_h
            || image_store->active_config.generation != hdr->generation
            || image_store->active_config.architecture_version != hdr->architecture_version
            || strcmp(image_store->active_config.model_name, hdr->model_name) != 0
            || strcmp(image_store->active_config.node_id, hdr->node_id) != 0) {
        weight_image_store_clear_locked(image_store);
    }
    image_store->active_config.state_publish_seq = hdr->publish_seq;
    image_store->active_config.generation = hdr->generation;
    image_store->active_config.architecture_version = hdr->architecture_version;
    image_store->active_config.round_id = hdr->round_id;
    image_store->active_config.cycle = hdr->cycle;
    image_store->active_config.step = hdr->step;
    image_store->active_config.mode = (int32_t)mode;
    image_store->active_config.target_w = target_w;
    image_store->active_config.target_h = target_h;
    image_store->active_config.render_w = render_w;
    image_store->active_config.render_h = render_h;
    image_store->active_config.render_c = 3;
    image_store->active_config.render_stride_bytes = render_w * 3;
    copy_cstr_trunc(image_store->active_config.model_name, sizeof(image_store->active_config.model_name), hdr->model_name);
    copy_cstr_trunc(image_store->active_config.node_id, sizeof(image_store->active_config.node_id), hdr->node_id);
    weight_image_store_unlock(image_store);
    rc = 0;

cleanup:
    if (shape0) free(shape0);
    if (numel) free(numel);
    if (data_ptrs) free(data_ptrs);
    if (names) free(names);
    mapped_blob_close(&blob);
    return rc;
}

NODUS_API int nodus_weight_image_store_get_active_config(
        const NodusWeightImageStore *store,
        uint64_t *out_state_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_mode,
        int32_t *out_target_w,
        int32_t *out_target_h,
        int32_t *out_render_w,
        int32_t *out_render_h,
        int32_t *out_render_c,
        int32_t *out_render_stride_bytes,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen)
{
    NodusWeightImageStore *st = (NodusWeightImageStore*)store;
    if (!st) return -1;
    weight_image_store_lock(st);
    if (st->active_config.state_publish_seq == 0u) {
        weight_image_store_unlock(st);
        return -1;
    }
    if (out_state_publish_seq) *out_state_publish_seq = st->active_config.state_publish_seq;
    if (out_generation) *out_generation = st->active_config.generation;
    if (out_architecture_version) *out_architecture_version = st->active_config.architecture_version;
    if (out_round_id) *out_round_id = st->active_config.round_id;
    if (out_cycle) *out_cycle = st->active_config.cycle;
    if (out_step) *out_step = st->active_config.step;
    if (out_mode) *out_mode = st->active_config.mode;
    if (out_target_w) *out_target_w = st->active_config.target_w;
    if (out_target_h) *out_target_h = st->active_config.target_h;
    if (out_render_w) *out_render_w = st->active_config.render_w;
    if (out_render_h) *out_render_h = st->active_config.render_h;
    if (out_render_c) *out_render_c = st->active_config.render_c;
    if (out_render_stride_bytes) *out_render_stride_bytes = st->active_config.render_stride_bytes;
    copy_out_string(st->active_config.model_name, out_model_name, out_model_name_buflen);
    copy_out_string(st->active_config.node_id, out_node_id, out_node_id_buflen);
    weight_image_store_unlock(st);
    return 0;
}

NODUS_API int nodus_weight_image_store_get_meta(
        const NodusWeightImageStore *store,
        int32_t index,
        uint64_t *out_image_seq,
        uint64_t *out_state_publish_seq,
        uint64_t *out_generation,
        uint64_t *out_architecture_version,
        int32_t *out_round_id,
        int32_t *out_cycle,
        int32_t *out_step,
        int32_t *out_w,
        int32_t *out_h,
        int32_t *out_c,
        int32_t *out_stride_bytes,
        int32_t *out_mode,
        int32_t *out_target_w,
        int32_t *out_target_h,
        uint64_t *out_byte_count,
        uint32_t *out_flags,
        char *out_model_name,
        int out_model_name_buflen,
        char *out_node_id,
        int out_node_id_buflen,
        char *out_blob_name,
        int out_blob_name_buflen)
{
    NodusWeightImageStore *st = (NodusWeightImageStore*)store;
    const NodusWeightImageEntry *entry;
    if (!st) return -1;
    weight_image_store_lock(st);
    if (index < 0 || index >= st->entry_count) {
        weight_image_store_unlock(st);
        return -1;
    }
    entry = &st->entries[index];
    if (entry->image_seq == 0u || entry->blob_name[0] == '\0') {
        weight_image_store_unlock(st);
        return -1;
    }
    if (out_image_seq) *out_image_seq = entry->image_seq;
    if (out_state_publish_seq) *out_state_publish_seq = entry->state_publish_seq;
    if (out_generation) *out_generation = entry->generation;
    if (out_architecture_version) *out_architecture_version = entry->architecture_version;
    if (out_round_id) *out_round_id = entry->round_id;
    if (out_cycle) *out_cycle = entry->cycle;
    if (out_step) *out_step = entry->step;
    if (out_w) *out_w = entry->w;
    if (out_h) *out_h = entry->h;
    if (out_c) *out_c = entry->c;
    if (out_stride_bytes) *out_stride_bytes = entry->stride_bytes;
    if (out_mode) *out_mode = entry->mode;
    if (out_target_w) *out_target_w = entry->target_w;
    if (out_target_h) *out_target_h = entry->target_h;
    if (out_byte_count) *out_byte_count = entry->byte_count;
    if (out_flags) *out_flags = entry->flags;
    copy_out_string(entry->model_name, out_model_name, out_model_name_buflen);
    copy_out_string(entry->node_id, out_node_id, out_node_id_buflen);
    copy_out_string(entry->blob_name, out_blob_name, out_blob_name_buflen);
    weight_image_store_unlock(st);
    return 0;
}

NODUS_API int32_t nodus_weight_image_store_copy_image(
        const NodusWeightImageStore *store,
        int32_t index,
        uint8_t *out_buf,
        uint64_t buf_size)
{
    uint64_t byte_count = 0;
    char blob_name[NODUS_WEIGHT_STORE_BLOB_NAME_MAX];
    NodusMappedBlobHandle blob = {0};
    if (!store || !out_buf || index < 0 || index >= store->entry_count) return -1;
    if (nodus_weight_image_store_get_meta(
            store,
            index,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            &byte_count,
            NULL,
            NULL,
            0,
            NULL,
            0,
            blob_name,
            (int)sizeof(blob_name)
        ) != 0) {
        return -1;
    }
    if (buf_size < byte_count) return -1;
    if (mapped_blob_open_ro(&blob, blob_name, byte_count) != 0) return -1;
    memcpy(out_buf, blob.ptr, (size_t)byte_count);
    mapped_blob_close(&blob);
    return (int32_t)byte_count;
}

NODUS_API int nodus_weight_image_store_mark_checkpoint(
        NodusWeightImageStore *store,
        uint64_t state_publish_seq,
        int32_t round_id,
        int32_t cycle)
{
    int logical;
    if (!store || state_publish_seq == 0u) return -1;
    weight_image_store_lock(store);
    for (logical = store->entry_count - 1; logical >= 0; logical--) {
        NodusWeightImageEntry *entry = &store->entries[logical];
        if (entry->state_publish_seq == state_publish_seq) {
            entry->flags |= NODUS_WEIGHT_IMAGE_FLAG_CHECKPOINT;
            entry->round_id = round_id;
            entry->cycle = cycle;
            weight_image_store_unlock(store);
            return 0;
        }
    }
    weight_image_store_unlock(store);
    return -1;
}

/* ================================================================== */
/*  Runtime Control Store                                             */
/* ================================================================== */

#define NODUS_RUNTIME_CONTROL_SHM_NAME   "NodusRuntimeControlStore_v1"
#define NODUS_RUNTIME_CONTROL_MTX_NAME   "NodusRuntimeControlStoreMutex_v1"
#define NODUS_RUNTIME_CONTROL_INIT_MAGIC 0x4e525443u

struct NodusRuntimeControlStore {
    uint32_t initialized;
    nodus_mutex_t lock;
    int32_t  active_service_count;
    uint64_t service_enter_count;
    uint64_t service_exit_count;
    int32_t  exit_requested;
    double   last_service_ts;
    double   exit_ts;
    char     last_source[NODUS_RUNTIME_CONTROL_SOURCE_MAX];
    char     exit_reason[NODUS_RUNTIME_CONTROL_REASON_MAX];
};

static NodusRuntimeControlStore *g_runtime_control_global = NULL;
#ifdef _WIN32
static HANDLE g_runtime_control_shm_mutex = NULL;
#endif

static void runtime_control_store_zero_fields(NodusRuntimeControlStore *store) {
    if (!store) return;
    store->active_service_count = 0;
    store->service_enter_count = 0u;
    store->service_exit_count = 0u;
    store->exit_requested = 0;
    store->last_service_ts = 0.0;
    store->exit_ts = 0.0;
    memset(store->last_source, 0, sizeof(store->last_source));
    memset(store->exit_reason, 0, sizeof(store->exit_reason));
}

static void runtime_control_store_init(NodusRuntimeControlStore *store, int cross_process) {
#ifdef _WIN32
    (void)cross_process;
    mutex_init(&store->lock);
#else
    if (cross_process) mutex_init_shared(&store->lock);
    else mutex_init(&store->lock);
#endif
    store->initialized = NODUS_RUNTIME_CONTROL_INIT_MAGIC;
    runtime_control_store_zero_fields(store);
}

#ifdef _WIN32
static void runtime_control_store_lock(NodusRuntimeControlStore *store) {
    if (store == g_runtime_control_global && g_runtime_control_shm_mutex)
        WaitForSingleObject(g_runtime_control_shm_mutex, INFINITE);
    else
        EnterCriticalSection(&store->lock);
}
static void runtime_control_store_unlock(NodusRuntimeControlStore *store) {
    if (store == g_runtime_control_global && g_runtime_control_shm_mutex)
        ReleaseMutex(g_runtime_control_shm_mutex);
    else
        LeaveCriticalSection(&store->lock);
}
#else
static void runtime_control_store_lock(NodusRuntimeControlStore *store)   { pthread_mutex_lock(&store->lock); }
static void runtime_control_store_unlock(NodusRuntimeControlStore *store) { pthread_mutex_unlock(&store->lock); }
#endif

NODUS_API NodusRuntimeControlStore* nodus_runtime_control_store_get_global(void) {
    if (g_runtime_control_global) return g_runtime_control_global;
#ifdef _WIN32
    {
        HANDLE shm = NULL;
        int created = 0;
        g_runtime_control_shm_mutex = CreateMutexA(NULL, FALSE, NODUS_RUNTIME_CONTROL_MTX_NAME);
        if (!g_runtime_control_shm_mutex) {
            fprintf(stderr, "[nodus] FATAL: CreateMutexA runtime control failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        shm = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
            (DWORD)sizeof(NodusRuntimeControlStore),
            NODUS_RUNTIME_CONTROL_SHM_NAME
        );
        if (!shm) {
            fprintf(stderr, "[nodus] FATAL: CreateFileMappingA runtime control failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        created = (GetLastError() != ERROR_ALREADY_EXISTS);
        g_runtime_control_global = (NodusRuntimeControlStore*)MapViewOfFile(
            shm, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NodusRuntimeControlStore));
        if (!g_runtime_control_global) {
            fprintf(stderr, "[nodus] FATAL: MapViewOfFile runtime control failed (%lu)\n", (unsigned long)GetLastError());
            abort();
        }
        if (created || g_runtime_control_global->initialized != NODUS_RUNTIME_CONTROL_INIT_MAGIC) {
            memset(g_runtime_control_global, 0, sizeof(NodusRuntimeControlStore));
            runtime_control_store_init(g_runtime_control_global, 1);
        }
    }
#else
    {
        int fd = shm_open("/" NODUS_RUNTIME_CONTROL_SHM_NAME, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            perror("[nodus] FATAL: shm_open runtime control");
            abort();
        }
        if (ftruncate(fd, (off_t)sizeof(NodusRuntimeControlStore)) != 0) {
            perror("[nodus] FATAL: ftruncate runtime control");
            abort();
        }
        g_runtime_control_global = (NodusRuntimeControlStore*)mmap(
            NULL, sizeof(NodusRuntimeControlStore), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (g_runtime_control_global == MAP_FAILED) {
            perror("[nodus] FATAL: mmap runtime control");
            abort();
        }
        close(fd);
        if (g_runtime_control_global->initialized != NODUS_RUNTIME_CONTROL_INIT_MAGIC) {
            memset(g_runtime_control_global, 0, sizeof(NodusRuntimeControlStore));
            runtime_control_store_init(g_runtime_control_global, 1);
        }
    }
#endif
    return g_runtime_control_global;
}

NODUS_API void nodus_runtime_control_store_clear(NodusRuntimeControlStore *store) {
    if (!store) return;
    runtime_control_store_lock(store);
    runtime_control_store_zero_fields(store);
    runtime_control_store_unlock(store);
}

NODUS_API void nodus_runtime_control_store_begin_service(
        NodusRuntimeControlStore *store,
        const char *source,
        double ts)
{
    if (!store) return;
    runtime_control_store_lock(store);
    if (store->active_service_count < INT32_MAX) {
        store->active_service_count += 1;
    }
    store->service_enter_count += 1u;
    store->last_service_ts = ts;
    copy_cstr_trunc(store->last_source, sizeof(store->last_source), source ? source : "");
    runtime_control_store_unlock(store);
}

NODUS_API void nodus_runtime_control_store_end_service(
        NodusRuntimeControlStore *store,
        const char *source,
        double ts)
{
    if (!store) return;
    runtime_control_store_lock(store);
    if (store->active_service_count > 0) {
        store->active_service_count -= 1;
    }
    store->service_exit_count += 1u;
    store->last_service_ts = ts;
    copy_cstr_trunc(store->last_source, sizeof(store->last_source), source ? source : "");
    runtime_control_store_unlock(store);
}

NODUS_API int32_t nodus_runtime_control_store_active_services(
        const NodusRuntimeControlStore *store)
{
    int32_t count = 0;
    NodusRuntimeControlStore *st = (NodusRuntimeControlStore*)store;
    if (!st) return 0;
    runtime_control_store_lock(st);
    count = st->active_service_count;
    runtime_control_store_unlock(st);
    return count;
}

NODUS_API void nodus_runtime_control_store_set_exit(
        NodusRuntimeControlStore *store,
        int32_t requested,
        const char *reason,
        double ts)
{
    if (!store) return;
    runtime_control_store_lock(store);
    store->exit_requested = requested ? 1 : 0;
    store->exit_ts = requested ? ts : 0.0;
    copy_cstr_trunc(store->exit_reason, sizeof(store->exit_reason),
                    requested ? (reason ? reason : "") : "");
    runtime_control_store_unlock(store);
}

NODUS_API int nodus_runtime_control_store_get_state(
        const NodusRuntimeControlStore *store,
        int32_t *out_active_service_count,
        uint64_t *out_service_enter_count,
        uint64_t *out_service_exit_count,
        int32_t *out_exit_requested,
        double *out_last_service_ts,
        double *out_exit_ts,
        char *out_last_source,
        int out_last_source_buflen,
        char *out_exit_reason,
        int out_exit_reason_buflen)
{
    NodusRuntimeControlStore *st = (NodusRuntimeControlStore*)store;
    if (!st) return -1;
    runtime_control_store_lock(st);
    if (out_active_service_count) *out_active_service_count = st->active_service_count;
    if (out_service_enter_count) *out_service_enter_count = st->service_enter_count;
    if (out_service_exit_count) *out_service_exit_count = st->service_exit_count;
    if (out_exit_requested) *out_exit_requested = st->exit_requested;
    if (out_last_service_ts) *out_last_service_ts = st->last_service_ts;
    if (out_exit_ts) *out_exit_ts = st->exit_ts;
    copy_out_string(st->last_source, out_last_source, out_last_source_buflen);
    copy_out_string(st->exit_reason, out_exit_reason, out_exit_reason_buflen);
    runtime_control_store_unlock(st);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Exported lock/unlock for stores that lacked public lock APIs      */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_weight_state_store_lock(NodusWeightStateStore *store) {
    if (store) weight_state_store_lock(store);
}
NODUS_API void nodus_weight_state_store_unlock(NodusWeightStateStore *store) {
    if (store) weight_state_store_unlock(store);
}
NODUS_API void nodus_weight_image_store_lock(NodusWeightImageStore *store) {
    if (store) weight_image_store_lock(store);
}
NODUS_API void nodus_weight_image_store_unlock(NodusWeightImageStore *store) {
    if (store) weight_image_store_unlock(store);
}
NODUS_API void nodus_runtime_control_store_lock(NodusRuntimeControlStore *store) {
    if (store) runtime_control_store_lock(store);
}
NODUS_API void nodus_runtime_control_store_unlock(NodusRuntimeControlStore *store) {
    if (store) runtime_control_store_unlock(store);
}
