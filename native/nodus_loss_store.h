/*
 * nodus_loss_store.h -- Cross-process loss data repository with locked access.
 *
 * Exports accessors for a flat memory store of per-channel loss records.
 * Built as a shared library (DLL / .so) consumed via Python ctypes.
 */

#ifndef NODUS_LOSS_STORE_H
#define NODUS_LOSS_STORE_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef NODUS_BUILDING_DLL
#    define NODUS_API __declspec(dllexport)
#  else
#    define NODUS_API __declspec(dllimport)
#  endif
#else
#  define NODUS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

#define NODUS_MAX_CHANNELS      64
#define NODUS_MAX_CHANNEL_NAME  64
#define NODUS_MAX_RECORDS       100000

/* ------------------------------------------------------------------ */
/*  Data types                                                        */
/* ------------------------------------------------------------------ */

/* Single loss observation -- 24 bytes. */
typedef struct NodusLossRecord {
    int32_t  step;
    int32_t  round_id;
    float    loss;
    float    aux;
    double   ts;
} NodusLossRecord;

/* Opaque handle to the store. */
typedef struct NodusLossStore NodusLossStore;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/* Create a new store.  max_channels clamped to NODUS_MAX_CHANNELS,
   max_records_per_channel clamped to NODUS_MAX_RECORDS.
   Returns NULL on allocation failure. */
NODUS_API NodusLossStore* nodus_loss_store_create(
        int max_channels,
        int max_records_per_channel);

/* Destroy the store and free all memory.  Safe to pass NULL. */
NODUS_API void nodus_loss_store_destroy(NodusLossStore *store);

/* ------------------------------------------------------------------ */
/*  Write                                                             */
/* ------------------------------------------------------------------ */

/* Record a loss value into the named channel.
   channel_key: UTF-8, max NODUS_MAX_CHANNEL_NAME-1 bytes.
   Returns the step index assigned, or -1 on error (e.g. store full of
   channels and this key is new). */
NODUS_API int32_t nodus_loss_store_record(
        NodusLossStore *store,
        const char     *channel_key,
        float           loss,
        float           aux,
        int32_t         round_id,
        double          ts);

/* Clear all channels and records. */
NODUS_API void nodus_loss_store_clear(NodusLossStore *store);

/* Clear a single channel. Returns 0 on success, -1 if not found. */
NODUS_API int nodus_loss_store_clear_channel(
        NodusLossStore *store,
        const char     *channel_key);

/* ------------------------------------------------------------------ */
/*  Read -- channel enumeration                                        */
/* ------------------------------------------------------------------ */

/* Number of active channels. */
NODUS_API int nodus_loss_store_channel_count(const NodusLossStore *store);

/* Copy the name of the i-th channel into buf (up to buf_len bytes).
   Returns the actual length, or -1 if index out of range. */
NODUS_API int nodus_loss_store_channel_name(
        const NodusLossStore *store,
        int    index,
        char  *buf,
        int    buf_len);

/* ------------------------------------------------------------------ */
/*  Read -- per-channel queries                                        */
/* ------------------------------------------------------------------ */

/* Total number of records stored for channel_key.  0 if not found. */
NODUS_API int32_t nodus_loss_store_channel_length(
        const NodusLossStore *store,
        const char           *channel_key);

/* Current step cursor (next step that will be assigned). */
NODUS_API int32_t nodus_loss_store_channel_cursor(
        const NodusLossStore *store,
        const char           *channel_key);

/* Copy up to max_out records with step >= from_step into out_buf.
   Returns number of records copied.  -1 if channel not found. */
NODUS_API int32_t nodus_loss_store_query_since(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        NodusLossRecord      *out_buf,
        int32_t               max_out);

/* Copy the latest record for a channel into *out.
   Returns 0 on success, -1 if channel empty or not found. */
NODUS_API int nodus_loss_store_latest(
        const NodusLossStore *store,
        const char           *channel_key,
        NodusLossRecord      *out);

/* Direct read-only pointer to the internal record array for a channel
   and its current length.  The pointer is valid until the next write
   to that channel or until the store is destroyed.
   Returns 0 on success, -1 if channel not found.
   Caller must hold no assumption about thread safety -- use
   nodus_loss_store_lock / _unlock around reads of this pointer. */
NODUS_API int nodus_loss_store_channel_data_ptr(
        const NodusLossStore  *store,
        const char            *channel_key,
        const NodusLossRecord **out_ptr,
        int32_t               *out_length);

/* ------------------------------------------------------------------ */
/*  Graph-line renderer                                               */
/* ------------------------------------------------------------------ */

/* Configuration for rendering a loss channel into an RGBA overlay line. */
typedef struct NodusGraphLineConfig {
    int32_t  plot_x0;       /* left pixel of plot area    */
    int32_t  plot_y0;       /* top pixel of plot area     */
    int32_t  plot_w;        /* width of plot area (px)    */
    int32_t  plot_h;        /* height of plot area (px)   */
    float    y_min;         /* loss value at bottom edge  */
    float    y_max;         /* loss value at top edge     */
    double   t_min;         /* wall-clock at left edge    */
    double   t_max;         /* wall-clock at right edge   */
    uint8_t  r, g, b, a;   /* line colour RGBA           */
    int32_t  from_step;     /* skip records before this step */
    int32_t  use_time_axis; /* 1 to use ts for x, 0 to use step index */
} NodusGraphLineConfig;

/* Render a single channel's loss series into an RGBA overlay image.
   out_rgba must point to plot_w * plot_h * 4 bytes of pre-zeroed memory.
   The function draws an anti-aliased 1px line in the specified colour on
   a fully-transparent background (alpha-isolated, ready for compositing).
   Returns the number of points plotted, or -1 on error. */
NODUS_API int32_t nodus_loss_store_render_graph_line(
        const NodusLossStore      *store,
        const char                *channel_key,
        const NodusGraphLineConfig *cfg,
        uint8_t                   *out_rgba);

/* Convenience: compute the Y-range (min, max) across all visible finite
   loss values in a channel, with 5% padding.  Returns 0 on success. */
NODUS_API int nodus_loss_store_channel_y_range(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        float                *out_min,
        float                *out_max);

/* Compute global time range across all channels.  Returns 0 on success. */
NODUS_API int nodus_loss_store_global_time_range(
        const NodusLossStore *store,
        double               *out_t_min,
        double               *out_t_max);

/* Copy `loss` field values into a caller-supplied float array.
   Returns count written, or -1 if channel not found.
   Records are in logical order (oldest first). */
NODUS_API int32_t nodus_loss_store_get_loss_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        float                *out_buf,
        int32_t               max_out);

/* Copy `ts` field values into a caller-supplied double array. */
NODUS_API int32_t nodus_loss_store_get_ts_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        double               *out_buf,
        int32_t               max_out);

/* Copy `step` field values into a caller-supplied int32 array. */
NODUS_API int32_t nodus_loss_store_get_step_array(
        const NodusLossStore *store,
        const char           *channel_key,
        int32_t               from_step,
        int32_t              *out_buf,
        int32_t               max_out);

/* ------------------------------------------------------------------ */
/*  Batch graph renderer                                              */
/* ------------------------------------------------------------------ */

/* Render multiple channels onto a single shared RGBA overlay in one call.
   The C side computes y_range and time_range internally; the caller only
   supplies channel keys, colors, and plot dimensions.

   channel_keys:  array of num_channels NUL-terminated UTF-8 key pointers.
   colors:        packed RGBA bytes, 4 * num_channels.
   display_start_frac: fraction [0,1) of each channel's history to skip
                       from the left (display trimming).
   out_rgba:      pre-zeroed buffer of plot_w * plot_h * 4 bytes.
   out_y_min / out_y_max / out_t_min / out_t_max: computed ranges
                  written back so the caller can draw gridlines / labels.

   Returns the total number of line segments drawn across all channels,
   or -1 on error. */
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
        double               *out_t_max);

/* ------------------------------------------------------------------ */
/*  Locking -- for callers that use channel_data_ptr                  */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_loss_store_lock(NodusLossStore *store);
NODUS_API void nodus_loss_store_unlock(NodusLossStore *store);

#ifdef __cplusplus
}
#endif

#endif /* NODUS_LOSS_STORE_H */
