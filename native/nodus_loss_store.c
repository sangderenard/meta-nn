/*
 * nodus_loss_store.c -- Implementation of the loss data repository.
 *
 * Single-process in-memory store with mutex-protected access.
 * Cross-platform: Windows (CRITICAL_SECTION) / POSIX (pthread_mutex).
 */

#ifndef NODUS_BUILDING_DLL
#  define NODUS_BUILDING_DLL
#endif
#include "nodus_loss_store.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   typedef CRITICAL_SECTION nodus_mutex_t;
   static void mutex_init(nodus_mutex_t *m)    { InitializeCriticalSection(m); }
   static void mutex_destroy(nodus_mutex_t *m) { DeleteCriticalSection(m); }
   static void mutex_lock(nodus_mutex_t *m)    { EnterCriticalSection(m); }
   static void mutex_unlock(nodus_mutex_t *m)  { LeaveCriticalSection(m); }
#else
#  include <pthread.h>
   typedef pthread_mutex_t nodus_mutex_t;
   static void mutex_init(nodus_mutex_t *m)    { pthread_mutex_init(m, NULL); }
   static void mutex_destroy(nodus_mutex_t *m) { pthread_mutex_destroy(m); }
   static void mutex_lock(nodus_mutex_t *m)    { pthread_mutex_lock(m); }
   static void mutex_unlock(nodus_mutex_t *m)  { pthread_mutex_unlock(m); }
#endif

/* ------------------------------------------------------------------ */
/*  Internal channel structure                                        */
/* ------------------------------------------------------------------ */

typedef struct NodusChannel {
    char             name[NODUS_MAX_CHANNEL_NAME];
    NodusLossRecord *records;       /* heap-allocated array */
    int32_t          capacity;      /* max_records_per_channel */
    int32_t          length;        /* current count */
    int32_t          step_cursor;   /* next step to assign */
    int32_t          head;          /* index of oldest record in circular buffer */
} NodusChannel;

struct NodusLossStore {
    nodus_mutex_t   lock;
    int             max_channels;
    int             max_records;
    int             channel_count;
    NodusChannel   *channels;       /* array of max_channels */
};

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

NODUS_API NodusLossStore* nodus_loss_store_create(
        int max_channels, int max_records_per_channel)
{
    max_channels           = clamp_i(max_channels, 1, NODUS_MAX_CHANNELS);
    max_records_per_channel = clamp_i(max_records_per_channel, 1, NODUS_MAX_RECORDS);

    NodusLossStore *store = (NodusLossStore*)calloc(1, sizeof(NodusLossStore));
    if (!store) return NULL;

    store->channels = (NodusChannel*)calloc((size_t)max_channels, sizeof(NodusChannel));
    if (!store->channels) {
        free(store);
        return NULL;
    }

    store->max_channels  = max_channels;
    store->max_records   = max_records_per_channel;
    store->channel_count = 0;

    /* Pre-allocate record arrays for every channel slot. */
    for (int i = 0; i < max_channels; i++) {
        store->channels[i].records =
            (NodusLossRecord*)calloc((size_t)max_records_per_channel,
                                     sizeof(NodusLossRecord));
        if (!store->channels[i].records) {
            /* Cleanup on failure. */
            for (int j = 0; j < i; j++)
                free(store->channels[j].records);
            free(store->channels);
            free(store);
            return NULL;
        }
        store->channels[i].capacity    = max_records_per_channel;
        store->channels[i].length      = 0;
        store->channels[i].step_cursor = 0;
        store->channels[i].head        = 0;
        store->channels[i].name[0]     = '\0';
    }

    mutex_init(&store->lock);
    return store;
}

NODUS_API void nodus_loss_store_destroy(NodusLossStore *store) {
    if (!store) return;
    mutex_destroy(&store->lock);
    if (store->channels) {
        for (int i = 0; i < store->max_channels; i++)
            free(store->channels[i].records);
        free(store->channels);
    }
    free(store);
}

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

    mutex_lock(&store->lock);

    int idx = find_channel(store, channel_key);
    if (idx < 0) {
        /* Allocate a new channel slot. */
        if (store->channel_count >= store->max_channels) {
            mutex_unlock(&store->lock);
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

    mutex_unlock(&store->lock);
    return step;
}

NODUS_API void nodus_loss_store_clear(NodusLossStore *store) {
    if (!store) return;
    mutex_lock(&store->lock);
    for (int i = 0; i < store->channel_count; i++) {
        store->channels[i].length      = 0;
        store->channels[i].step_cursor = 0;
        store->channels[i].head        = 0;
        store->channels[i].name[0]     = '\0';
    }
    store->channel_count = 0;
    mutex_unlock(&store->lock);
}

NODUS_API int nodus_loss_store_clear_channel(
        NodusLossStore *store, const char *channel_key)
{
    if (!store || !channel_key) return -1;
    mutex_lock(&store->lock);
    int idx = find_channel(store, channel_key);
    if (idx < 0) {
        mutex_unlock(&store->lock);
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
        /* Swap record pointers -- don't free, just reuse the slot's array. */
        NodusLossRecord *tmp = ch->records;
        store->channels[idx].records = store->channels[last].records;
        store->channels[last].records = tmp;
    }
    store->channels[last].name[0] = '\0';
    store->channels[last].length  = 0;
    store->channel_count--;
    mutex_unlock(&store->lock);
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
    if (store) mutex_lock(&store->lock);
}

NODUS_API void nodus_loss_store_unlock(NodusLossStore *store) {
    if (store) mutex_unlock(&store->lock);
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

    mutex_lock((nodus_mutex_t*)&((NodusLossStore*)store)->lock);

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
        mutex_unlock((nodus_mutex_t*)&((NodusLossStore*)store)->lock);
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

    mutex_unlock((nodus_mutex_t*)&((NodusLossStore*)store)->lock);
    return total_segments;
}
