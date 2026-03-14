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
   typedef pthread_mutex_t nodus_mutex_t;

   static void mutex_init(nodus_mutex_t *m)    { pthread_mutex_init(m, NULL); }
   static void mutex_init_shared(nodus_mutex_t *m) {
       pthread_mutexattr_t attr;
       pthread_mutexattr_init(&attr);
       pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
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
    return store ? store->channel_count : 0;
}

NODUS_API int nodus_loss_store_channel_name(
        const NodusLossStore *store, int index, char *buf, int buf_len)
{
    if (!store || !buf || buf_len <= 0) return -1;
    if (index < 0 || index >= store->channel_count) return -1;
    int len = (int)strlen(store->channels[index].name);
    int copy = (len < buf_len - 1) ? len : (buf_len - 1);
    memcpy(buf, store->channels[index].name, (size_t)copy);
    buf[copy] = '\0';
    return len;
}

/* ------------------------------------------------------------------ */
/*  Read -- per-channel queries                                        */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_loss_store_channel_length(
        const NodusLossStore *store, const char *channel_key)
{
    if (!store || !channel_key) return 0;
    /* Cast away const for find_channel -- it only reads. */
    int idx = find_channel(store, channel_key);
    return (idx >= 0) ? store->channels[idx].length : 0;
}

NODUS_API int32_t nodus_loss_store_channel_cursor(
        const NodusLossStore *store, const char *channel_key)
{
    if (!store || !channel_key) return 0;
    int idx = find_channel(store, channel_key);
    return (idx >= 0) ? store->channels[idx].step_cursor : 0;
}

NODUS_API int32_t nodus_loss_store_query_since(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        NodusLossRecord      *out_buf,
        int32_t               max_out)
{
    if (!store || !channel_key || !out_buf || max_out <= 0) return -1;

    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;

    const NodusChannel *ch = &store->channels[idx];
    int32_t copied = 0;

    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step) {
            out_buf[copied++] = ch->records[bi];
        }
    }
    return copied;
}

NODUS_API int nodus_loss_store_latest(
        const NodusLossStore *store,
        const char           *channel_key,
        NodusLossRecord      *out)
{
    if (!store || !channel_key || !out) return -1;

    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;

    const NodusChannel *ch = &store->channels[idx];
    if (ch->length == 0) return -1;

    /* Latest is at logical position length-1. */
    int bi = buf_index(ch, ch->length - 1);
    *out = ch->records[bi];
    return 0;
}

NODUS_API int nodus_loss_store_channel_data_ptr(
        const NodusLossStore  *store,
        const char            *channel_key,
        const NodusLossRecord **out_ptr,
        int32_t               *out_length)
{
    if (!store || !channel_key || !out_ptr || !out_length) return -1;

    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;

    const NodusChannel *ch = &store->channels[idx];
    *out_ptr   = ch->records;
    *out_length = ch->length;
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
    if (!store || !channel_key || !out_buf || max_out <= 0) return -1;
    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;
    const NodusChannel *ch = &store->channels[idx];
    int32_t copied = 0;
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].loss;
    }
    return copied;
}

NODUS_API int32_t nodus_loss_store_get_ts_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        double               *out_buf,
        int32_t               max_out)
{
    if (!store || !channel_key || !out_buf || max_out <= 0) return -1;
    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;
    const NodusChannel *ch = &store->channels[idx];
    int32_t copied = 0;
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].ts;
    }
    return copied;
}

NODUS_API int32_t nodus_loss_store_get_step_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        int32_t              *out_buf,
        int32_t               max_out)
{
    if (!store || !channel_key || !out_buf || max_out <= 0) return -1;
    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;
    const NodusChannel *ch = &store->channels[idx];
    int32_t copied = 0;
    for (int32_t i = 0; i < ch->length && copied < max_out; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step)
            out_buf[copied++] = ch->records[bi].step;
    }
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
    if (!store || !channel_key || !out_min || !out_max) return -1;
    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;
    const NodusChannel *ch = &store->channels[idx];
    float lo =  1e30f;
    float hi = -1e30f;
    int count = 0;
    for (int32_t i = 0; i < ch->length; i++) {
        int bi = buf_index(ch, i);
        const NodusLossRecord *r = &ch->records[bi];
        if (r->step < from_step) continue;
        if (!isfinite(r->loss)) continue;
        if (r->loss < lo) lo = r->loss;
        if (r->loss > hi) hi = r->loss;
        count++;
    }
    if (count < 1) return -1;
    float span = hi - lo;
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
    if (!store || !out_t_min || !out_t_max) return -1;
    double lo = 1e30, hi = -1e30;
    int found = 0;
    for (int c = 0; c < store->channel_count; c++) {
        const NodusChannel *ch = &store->channels[c];
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
    if (!store || !channel_key || !cfg || !out_rgba) return -1;
    int idx = find_channel(store, channel_key);
    if (idx < 0) return -1;

    const NodusChannel *ch = &store->channels[idx];
    int32_t pw = cfg->plot_w;
    int32_t ph = cfg->plot_h;
    float ymin = cfg->y_min;
    float ymax = cfg->y_max;
    double tmin = cfg->t_min;
    double tmax = cfg->t_max;
    int use_time = cfg->use_time_axis;
    int32_t from_step = cfg->from_step;

    if (pw <= 0 || ph <= 0) return -1;
    if (ymax <= ymin + 1e-9f) return -1;

    /* Collect visible records. */
    int32_t n_visible = 0;
    for (int32_t i = 0; i < ch->length; i++) {
        int bi = buf_index(ch, i);
        if (ch->records[bi].step >= from_step) n_visible++;
    }
    if (n_visible < 2) return 0;

    /* We need to iterate twice -- first to count for downsampling, then to draw.
       For efficiency, compute x,y pixel pairs in a local stack if small,
       else heap. */
    int use_heap = (n_visible > 8192);
    int *px_buf = NULL;
    int stack_buf[8192 * 2];
    if (use_heap) {
        px_buf = (int*)malloc(sizeof(int) * (size_t)n_visible * 2);
        if (!px_buf) return -1;
    } else {
        px_buf = stack_buf;
    }

    /* Compute time span for step-index mode. */
    double t_span = (tmax > tmin + 1e-9) ? (tmax - tmin) : 1.0;
    /* For step-index mode, count visible for uniform spacing. */
    int32_t vi = 0;
    int32_t plotted = 0;
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

        /* Clamp to plot area. */
        if (xp < 0) xp = 0; if (xp >= pw) xp = pw - 1;
        if (yp < 0) yp = 0; if (yp >= ph) yp = ph - 1;

        px_buf[plotted * 2 + 0] = xp;
        px_buf[plotted * 2 + 1] = yp;
        plotted++;
        vi++;
    }

    /* Draw line segments. */
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

#define NODUS_SCRUB_SHM_NAME    "NodusScrubRingGlobal_v1"
#define NODUS_SCRUB_MTX_NAME    "NodusScrubRingMutex_v1"
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
    return ring ? ring->length : 0;
}

NODUS_API int32_t nodus_scrub_ring_capacity(const NodusScrubRing *ring) {
    return ring ? ring->capacity : 0;
}

NODUS_API int32_t nodus_scrub_ring_write_cursor(const NodusScrubRing *ring) {
    return ring ? ring->write_cursor : 0;
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
    if (!ring || index < 0 || index >= ring->length) return -1;

    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];

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

    return 0;
}

NODUS_API int32_t nodus_scrub_ring_copy_output_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    if (!ring || !out_buf || index < 0 || index >= ring->length) return -1;
    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_OUTPUT)) return 0;
    uint32_t img_bytes = e->output_w * e->output_h * NODUS_SCRUB_IMAGE_C;
    if (img_bytes > NODUS_SCRUB_IMAGE_BYTES) img_bytes = NODUS_SCRUB_IMAGE_BYTES;
    uint32_t n = (img_bytes < buf_size) ? img_bytes : buf_size;
    memcpy(out_buf, e->output_image, n);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_training_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    if (!ring || !out_buf || index < 0 || index >= ring->length) return -1;
    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_IMAGE)) return 0;
    uint32_t img_bytes = e->image_w * e->image_h * NODUS_SCRUB_IMAGE_C;
    if (img_bytes > NODUS_SCRUB_IMAGE_BYTES) img_bytes = NODUS_SCRUB_IMAGE_BYTES;
    uint32_t n = (img_bytes < buf_size) ? img_bytes : buf_size;
    memcpy(out_buf, e->training_image, n);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_target(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size)
{
    if (!ring || !out_buf || index < 0 || index >= ring->length) return -1;
    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_TARGET)) return 0;
    uint32_t n = (e->target_len < buf_size) ? e->target_len : buf_size;
    memcpy(out_buf, e->target_data, n);
    return (int32_t)n;
}

NODUS_API int32_t nodus_scrub_ring_copy_thumbnail(
        const NodusScrubRing *ring, int32_t index, int thumb_idx,
        uint8_t *out_buf, uint32_t buf_size)
{
    if (!ring || !out_buf || index < 0 || index >= ring->length) return -1;
    if (thumb_idx < 0 || thumb_idx >= NODUS_SCRUB_NUM_THUMBS) return -1;
    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];
    if (!(e->flags & NODUS_SCRUB_FLAG_HAS_THUMBS)) return 0;
    uint32_t n = (NODUS_SCRUB_THUMB_BYTES < buf_size)
               ? NODUS_SCRUB_THUMB_BYTES : buf_size;
    memcpy(out_buf, e->thumbnails[thumb_idx], n);
    return (int32_t)n;
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
   Box-average filter.  src and dst must NOT alias. */
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
            uint32_t acc_r = 0, acc_g = 0, acc_b = 0, count = 0;
            for (uint32_t sy = sy0; sy < sy1 && sy < src_h; sy++) {
                for (uint32_t sx = sx0; sx < sx1 && sx < src_w; sx++) {
                    uint32_t off = (sy * src_w + sx) * 4;
                    acc_r += src[off + 0];
                    acc_g += src[off + 1];
                    acc_b += src[off + 2];
                    count++;
                }
            }
            uint32_t dst_off = (dy * dst_w + dx) * 3;
            if (count > 0) {
                dst[dst_off + 0] = (uint8_t)(acc_r / count);
                dst[dst_off + 1] = (uint8_t)(acc_g / count);
                dst[dst_off + 2] = (uint8_t)(acc_b / count);
            }
        }
    }
}

NODUS_API int nodus_composite_build_frame(
        const NodusScrubRing *ring, int32_t index,
        uint32_t panel_w, uint32_t panel_h,
        NodusCompositeFrame *out_frame)
{
    if (!ring || !out_frame || index < 0 || index >= ring->length) return -1;
    if (panel_w == 0 || panel_h == 0) return -1;
    if (panel_w > NODUS_COMPOSITE_PANEL_MAX_W) panel_w = NODUS_COMPOSITE_PANEL_MAX_W;
    if (panel_h > NODUS_COMPOSITE_PANEL_MAX_H) panel_h = NODUS_COMPOSITE_PANEL_MAX_H;

    int bi = scrub_buf_index(ring, index);
    const NodusScrubEntry *e = &ring->entries[bi];

    out_frame->source_ring_cursor = ring->write_cursor;
    out_frame->step      = e->step;
    out_frame->round_id  = e->round_id;
    out_frame->ts        = e->ts;
    out_frame->loss      = e->loss;
    out_frame->panel_w   = panel_w;
    out_frame->panel_h   = panel_h;
    out_frame->flags     = e->flags;

    uint32_t src_w = e->image_w;
    uint32_t src_h = e->image_h;
    uint32_t out_w = e->output_w;
    uint32_t out_h = e->output_h;

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
