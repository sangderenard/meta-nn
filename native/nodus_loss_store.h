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

/* Return the process-global singleton store backed by OS shared memory.
   Every process that loads this DLL and calls this function receives the
   SAME physical memory.  The store is created on first call with the
   compile-time NODUS_MAX_CHANNELS / NODUS_MAX_RECORDS limits.
   Never returns NULL (aborts on allocation failure).
   Do NOT call nodus_loss_store_destroy on the returned pointer. */
NODUS_API NodusLossStore* nodus_loss_store_get_global(void);

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

/* ================================================================== */
/*  Scrub Ring -- cross-process training-image & frame cache          */
/* ================================================================== */

/* Full-resolution training image (RGBA). */
#define NODUS_SCRUB_IMAGE_W         256
#define NODUS_SCRUB_IMAGE_H         256
#define NODUS_SCRUB_IMAGE_C         4
#define NODUS_SCRUB_IMAGE_BYTES     (NODUS_SCRUB_IMAGE_W * NODUS_SCRUB_IMAGE_H * NODUS_SCRUB_IMAGE_C)

/* Target data buffer -- same capacity as training image (masks etc). */
#define NODUS_SCRUB_TARGET_BYTES    NODUS_SCRUB_IMAGE_BYTES

/* Display thumbnails (3 panels: target+mask, diff, detected+mask). */
#define NODUS_SCRUB_THUMB_W         64
#define NODUS_SCRUB_THUMB_H         64
#define NODUS_SCRUB_THUMB_C         3
#define NODUS_SCRUB_THUMB_BYTES     (NODUS_SCRUB_THUMB_W * NODUS_SCRUB_THUMB_H * NODUS_SCRUB_THUMB_C)
#define NODUS_SCRUB_NUM_THUMBS      3

/* Ring capacity: 512 blue-zone (gradient checkpoints) + 1024 amber. */
#define NODUS_SCRUB_RING_CAPACITY   1536
#define NODUS_SCRUB_BLUE_ZONE       512

/* Per-entry flags. */
#define NODUS_SCRUB_FLAG_HAS_IMAGE      0x01
#define NODUS_SCRUB_FLAG_HAS_TARGET     0x02
#define NODUS_SCRUB_FLAG_HAS_THUMBS     0x04
#define NODUS_SCRUB_FLAG_HAS_GRADIENT   0x08
#define NODUS_SCRUB_FLAG_CHECKPOINT     0x10
#define NODUS_SCRUB_FLAG_HAS_OUTPUT     0x20
#define NODUS_SCRUB_FLAG_REDUCED        0x40

/* Opaque handle. */
typedef struct NodusScrubRing NodusScrubRing;

/* ------------------------------------------------------------------ */
/*  Scrub Ring lifecycle                                              */
/* ------------------------------------------------------------------ */

NODUS_API NodusScrubRing* nodus_scrub_ring_get_global(void);
NODUS_API NodusScrubRing* nodus_scrub_ring_create(void);
NODUS_API void nodus_scrub_ring_destroy(NodusScrubRing *ring);

/* ------------------------------------------------------------------ */
/*  Scrub Ring write                                                  */
/* ------------------------------------------------------------------ */

/* Push a new entry into the ring.  Returns write cursor index, -1 on error.
   training_image: raw pixel bytes (RGB or RGBA), up to NODUS_SCRUB_IMAGE_BYTES.
                   If len == image_w*image_h*3 (RGB), auto-expands to RGBA.
   output_image:   same format/rules as training_image.
   target_data:    target/mask bytes (RGB or RGBA), up to NODUS_SCRUB_TARGET_BYTES.
                   If len == image_w*image_h*3 (RGB), auto-expands to RGBA.
   thumb0..thumb2: exactly NODUS_SCRUB_THUMB_BYTES each, or NULL to skip.
   The training backend should pass full-size images and ignore thumbnails;
   thumbnails are produced by the GUI side when it needs them. */
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
        const uint8_t *thumb2);

NODUS_API void nodus_scrub_ring_clear(NodusScrubRing *ring);

/* ------------------------------------------------------------------ */
/*  Scrub Ring read                                                   */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_scrub_ring_length(const NodusScrubRing *ring);
NODUS_API int32_t nodus_scrub_ring_capacity(const NodusScrubRing *ring);
NODUS_API int32_t nodus_scrub_ring_write_cursor(const NodusScrubRing *ring);

/* Metadata for entry at logical index (0 = oldest, length-1 = newest).
   Any output pointer may be NULL to skip that field. */
NODUS_API int nodus_scrub_ring_get_meta(
        const NodusScrubRing *ring, int32_t index,
        int32_t *out_step, int32_t *out_round_id, double *out_ts,
        float *out_loss, char *out_channel_key, int channel_key_buflen,
        uint32_t *out_flags,
        uint32_t *out_image_w, uint32_t *out_image_h,
        uint32_t *out_target_len,
        uint32_t *out_output_w, uint32_t *out_output_h);

/* Copy training image into caller buffer.  Returns bytes copied, -1 error. */
NODUS_API int32_t nodus_scrub_ring_copy_training_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size);

/* Copy target data into caller buffer.  Returns bytes copied, -1 error. */
NODUS_API int32_t nodus_scrub_ring_copy_target(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size);

/* Copy output image into caller buffer.  Returns bytes copied, -1 error. */
NODUS_API int32_t nodus_scrub_ring_copy_output_image(
        const NodusScrubRing *ring, int32_t index,
        uint8_t *out_buf, uint32_t buf_size);

/* Copy thumbnail.  thumb_idx in [0..2].  Returns bytes copied, -1 error. */
NODUS_API int32_t nodus_scrub_ring_copy_thumbnail(
        const NodusScrubRing *ring, int32_t index, int thumb_idx,
        uint8_t *out_buf, uint32_t buf_size);

/* ------------------------------------------------------------------ */
/*  Scrub Ring quality reduction (GUI-side cache management)          */
/* ------------------------------------------------------------------ */

/* Reduce the output image at `index` by box-averaging each target_w x target_h
   block, rewriting the stored pixels in-place and updating image_w/image_h.
   If target_w/target_h are 0, the output image buffer is cleared entirely.
   Sets NODUS_SCRUB_FLAG_REDUCED on the entry.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_scrub_ring_reduce_output(
        NodusScrubRing *ring, int32_t index,
        uint32_t target_w, uint32_t target_h);

/* Reduce the training image at `index` the same way.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_scrub_ring_reduce_training(
        NodusScrubRing *ring, int32_t index,
        uint32_t target_w, uint32_t target_h);

/* ------------------------------------------------------------------ */
/*  Scrub Ring locking                                                */
/* ------------------------------------------------------------------ */

NODUS_API void nodus_scrub_ring_lock(NodusScrubRing *ring);
NODUS_API void nodus_scrub_ring_unlock(NodusScrubRing *ring);

/* ================================================================== */
/*  Composite Frame Builder + GUI-side Cache                          */
/* ================================================================== */

/* Maximum composite panel dimensions (matches max scrub image size). */
#define NODUS_COMPOSITE_PANEL_MAX_W     256
#define NODUS_COMPOSITE_PANEL_MAX_H     256
#define NODUS_COMPOSITE_PANEL_C         3   /* RGB */
#define NODUS_COMPOSITE_PANEL_MAX_BYTES \
    (NODUS_COMPOSITE_PANEL_MAX_W * NODUS_COMPOSITE_PANEL_MAX_H * NODUS_COMPOSITE_PANEL_C)

/* Default GUI composite-history capacity. */
#define NODUS_COMPOSITE_CACHE_CAPACITY  512

/* A single GUI display frame: 3 RGB panels + metadata. */
typedef struct NodusCompositeFrame {
    int32_t  source_ring_cursor;
    int32_t  step;
    int32_t  round_id;
    double   ts;
    float    loss;
    uint32_t panel_w;
    uint32_t panel_h;
    uint32_t flags;
    uint8_t  target_panel[NODUS_COMPOSITE_PANEL_MAX_BYTES];
    uint8_t  input_panel[NODUS_COMPOSITE_PANEL_MAX_BYTES];
    uint8_t  output_panel[NODUS_COMPOSITE_PANEL_MAX_BYTES];
} NodusCompositeFrame;

/* Opaque handle for the local composite cache. */
typedef struct NodusCompositeCache NodusCompositeCache;

/* ------------------------------------------------------------------ */
/*  Composite building                                                */
/* ------------------------------------------------------------------ */

/* Build a composite frame from scrub ring entry at `index`.
   Down-samples each RGBA image to `panel_w x panel_h` RGB.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_composite_build_frame(
        const NodusScrubRing *ring, int32_t index,
        uint32_t panel_w, uint32_t panel_h,
        NodusCompositeFrame *out_frame);

/* Build a composite frame from scrub ring entry and push it directly
   into a composite cache in one call.
   Returns the cache write index, -1 on error. */
NODUS_API int32_t nodus_composite_build_and_push(
        const NodusScrubRing *ring, int32_t ring_index,
        uint32_t panel_w, uint32_t panel_h,
        NodusCompositeCache *cache);

/* ------------------------------------------------------------------ */
/*  Composite cache lifecycle                                         */
/* ------------------------------------------------------------------ */

NODUS_API NodusCompositeCache* nodus_composite_cache_create(int32_t capacity);
NODUS_API void nodus_composite_cache_destroy(NodusCompositeCache *cache);
NODUS_API void nodus_composite_cache_clear(NodusCompositeCache *cache);

/* ------------------------------------------------------------------ */
/*  Composite cache read/write                                        */
/* ------------------------------------------------------------------ */

NODUS_API int32_t nodus_composite_cache_push(
        NodusCompositeCache *cache,
        const NodusCompositeFrame *frame);

NODUS_API int32_t nodus_composite_cache_length(const NodusCompositeCache *cache);
NODUS_API int32_t nodus_composite_cache_capacity(const NodusCompositeCache *cache);

/* Metadata for composite frame at logical index (0 = oldest). */
NODUS_API int nodus_composite_cache_get_meta(
        const NodusCompositeCache *cache, int32_t index,
        int32_t *out_step, int32_t *out_round_id, double *out_ts,
        float *out_loss, uint32_t *out_flags,
        uint32_t *out_panel_w, uint32_t *out_panel_h,
        int32_t *out_source_ring_cursor);

/* Copy one RGB panel from a cached frame.
   panel_idx: 0 = target, 1 = input, 2 = output.
   out_buf must be >= panel_w * panel_h * 3.
   Returns bytes copied, -1 on error. */
NODUS_API int32_t nodus_composite_cache_copy_panel(
        const NodusCompositeCache *cache, int32_t index, int panel_idx,
        uint8_t *out_buf, uint32_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* NODUS_LOSS_STORE_H */
