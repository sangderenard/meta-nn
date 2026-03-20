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
#define NODUS_SCRUB_FLAG_HAS_TEXT       0x80

/* Frame text buffers — parallel to images, one set per entry. */
#define NODUS_SCRUB_NUM_PANELS          3
#define NODUS_SCRUB_CAPTION_BYTES       256   /* single caption line, null-terminated */
#define NODUS_SCRUB_TITLE_BYTES          64   /* per-panel title, null-terminated     */
#define NODUS_SCRUB_ROWS_BYTES         1024   /* per-panel rows, '\n'-delimited       */

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
/*  Scrub Ring text — parallel to images, same lock                  */
/* ------------------------------------------------------------------ */

/* Write text fields into the slot for the given cursor (value returned by
   nodus_scrub_ring_push).  All string pointers may be NULL to clear the field.
   Sets NODUS_SCRUB_FLAG_HAS_TEXT.  Safe to call from any thread. */
NODUS_API void nodus_scrub_ring_write_text(
        NodusScrubRing *ring, int32_t cursor,
        const char *caption,
        const char *title0,  const char *title1,  const char *title2,
        const char *rows0,   const char *rows1,   const char *rows2);

/* Read text fields from the slot for the given cursor.  Any out_* pointer
   may be NULL to skip that field.  Returns 1 if HAS_TEXT was set, 0 if the
   slot has no text, -1 on error.  Safe to call from any thread. */
NODUS_API int nodus_scrub_ring_read_text(
        const NodusScrubRing *ring, int32_t cursor,
        char *out_caption,  int caption_buf_size,
        char *out_title0,   int title0_buf_size,
        char *out_title1,   int title1_buf_size,
        char *out_title2,   int title2_buf_size,
        char *out_rows0,    int rows0_buf_size,
        char *out_rows1,    int rows1_buf_size,
        char *out_rows2,    int rows2_buf_size);

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

/* ================================================================== */
/*  Weight State Store + Rendered Image Cache                         */
/* ================================================================== */

#define NODUS_WEIGHT_STORE_NAME_MAX          64
#define NODUS_WEIGHT_STORE_BLOB_NAME_MAX     128
#define NODUS_WEIGHT_STATE_REGISTRY_CAPACITY 64
#define NODUS_WEIGHT_IMAGE_CACHE_CAPACITY    256
#define NODUS_WEIGHT_IMAGE_DEFAULT_MAX_BYTES ((uint64_t)256u * 1024u * 1024u)

#define NODUS_WEIGHT_IMAGE_FLAG_CHECKPOINT   0x01

typedef struct NodusWeightStateStore NodusWeightStateStore;
typedef struct NodusWeightImageStore NodusWeightImageStore;

/* ------------------------------------------------------------------ */
/*  Weight state store                                                */
/* ------------------------------------------------------------------ */

/* Global cross-process store that retains the latest published state-dict
   snapshot for the actively training model. The payload itself lives in a
   versioned shared-memory blob so it can resize when model shapes change. */
NODUS_API NodusWeightStateStore* nodus_weight_state_store_get_global(void);

/* Publish a flattened float32 state-dict into the cross-process store.
   The layout matches nodus_weight_image_from_state_dict(): each parameter is
   identified by name and described by a flat float pointer, element count,
   and shape0 (dimension-0 size).
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_weight_state_store_publish_flat(
        NodusWeightStateStore *store,
        const char            *model_name,
        const char            *node_id,
        int32_t                round_id,
        int32_t                cycle,
        int32_t                step,
        uint64_t               generation,
        uint64_t               architecture_version,
        uint64_t               publish_seq,
        const char *const     *param_names,
        const float *const    *param_data,
        const int32_t         *param_numel,
        const int32_t         *param_shape0,
        int32_t                num_params);

/* Read the latest published metadata. Any output pointer may be NULL.
   Returns 0 on success, -1 if no snapshot has been published yet. */
NODUS_API int nodus_weight_state_store_get_meta(
        const NodusWeightStateStore *store,
        uint64_t                    *out_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_param_count,
        uint64_t                    *out_blob_bytes,
        char                        *out_model_name,
        int                          out_model_name_buflen,
        char                        *out_node_id,
        int                          out_node_id_buflen,
        char                        *out_blob_name,
        int                          out_blob_name_buflen);

/* Enumerate the keyed latest-state registry. Index 0 = oldest retained model,
   count-1 = most recently published model entry. */
NODUS_API int32_t nodus_weight_state_store_count(const NodusWeightStateStore *store);

NODUS_API int nodus_weight_state_store_get_meta_at(
        const NodusWeightStateStore *store,
        int32_t                      index,
        uint64_t                    *out_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_param_count,
        uint64_t                    *out_blob_bytes,
        char                        *out_model_name,
        int                          out_model_name_buflen,
        char                        *out_node_id,
        int                          out_node_id_buflen,
        char                        *out_blob_name,
        int                          out_blob_name_buflen);

NODUS_API int nodus_weight_state_store_get_meta_for(
        const NodusWeightStateStore *store,
        const char                  *model_name,
        const char                  *node_id,
        uint64_t                    *out_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_param_count,
        uint64_t                    *out_blob_bytes,
        char                        *out_blob_name,
        int                          out_blob_name_buflen);

/* ------------------------------------------------------------------ */
/*  Weight image cache                                                */
/* ------------------------------------------------------------------ */

/* Global cross-process rendered-image cache. The GUI is the sole writer. */
NODUS_API NodusWeightImageStore* nodus_weight_image_store_get_global(void);

NODUS_API int32_t nodus_weight_image_store_length(const NodusWeightImageStore *store);
NODUS_API int32_t nodus_weight_image_store_capacity(const NodusWeightImageStore *store);

/* Set the effective image-cache limits.
   max_entries is clamped to [1, NODUS_WEIGHT_IMAGE_CACHE_CAPACITY].
   max_total_bytes == 0 selects NODUS_WEIGHT_IMAGE_DEFAULT_MAX_BYTES.
   Oldest non-checkpoint images are evicted first to satisfy the new limits.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_weight_image_store_set_limits(
        NodusWeightImageStore *store,
        int32_t                max_entries,
        uint64_t               max_total_bytes);

/* Read the current effective image-cache limits and usage. */
NODUS_API int nodus_weight_image_store_get_stats(
        const NodusWeightImageStore *store,
        int32_t                     *out_max_entries,
        int32_t                     *out_entry_count,
        uint64_t                    *out_max_total_bytes,
        uint64_t                    *out_total_bytes);

/* Configure the active GUI-owned render contract for the latest state snapshot.
   Measures the exact rendered size for (mode, target_w, target_h) and records
   that configuration in the image store control block.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_weight_image_store_configure_latest(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        int                    mode,
        int32_t                target_w,
        int32_t                target_h);

/* Read the active image configuration. Returns 0 on success, -1 if unset. */
NODUS_API int nodus_weight_image_store_get_active_config(
        const NodusWeightImageStore *store,
        uint64_t                    *out_state_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_mode,
        int32_t                     *out_target_w,
        int32_t                     *out_target_h,
        int32_t                     *out_render_w,
        int32_t                     *out_render_h,
        int32_t                     *out_render_c,
        int32_t                     *out_render_stride_bytes,
        char                        *out_model_name,
        int                          out_model_name_buflen,
        char                        *out_node_id,
        int                          out_node_id_buflen);

/* Render the latest published weight snapshot into the image cache.
   The state store is read, rendered entirely in C, and the resulting RGB
   image is published into the image cache as the newest entry.
   Returns 0 on success, -1 on error. */
NODUS_API int nodus_weight_image_store_render_latest(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        int                    mode,
        int32_t                target_w,
        int32_t                target_h);

/* Measure/render a specific keyed model entry from the multi-model state store. */
NODUS_API int nodus_weight_image_store_measure_for(
        const NodusWeightStateStore *state_store,
        const char                  *model_name,
        const char                  *node_id,
        int                          mode,
        int32_t                      target_w,
        int32_t                      target_h,
        uint64_t                    *out_state_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_render_w,
        int32_t                     *out_render_h,
        int32_t                     *out_render_c,
        int32_t                     *out_render_stride_bytes);

NODUS_API int nodus_weight_image_store_render_for(
        NodusWeightStateStore *state_store,
        NodusWeightImageStore *image_store,
        const char            *model_name,
        const char            *node_id,
        int                    mode,
        int32_t                target_w,
        int32_t                target_h);

/* Read image-entry metadata at logical index (0 = oldest, length-1 = newest).
   Any output pointer may be NULL. Returns 0 on success, -1 on error. */
NODUS_API int nodus_weight_image_store_get_meta(
        const NodusWeightImageStore *store,
        int32_t                      index,
        uint64_t                    *out_image_seq,
        uint64_t                    *out_state_publish_seq,
        uint64_t                    *out_generation,
        uint64_t                    *out_architecture_version,
        int32_t                     *out_round_id,
        int32_t                     *out_cycle,
        int32_t                     *out_step,
        int32_t                     *out_w,
        int32_t                     *out_h,
        int32_t                     *out_c,
        int32_t                     *out_stride_bytes,
        int32_t                     *out_mode,
        int32_t                     *out_target_w,
        int32_t                     *out_target_h,
        uint64_t                    *out_byte_count,
        uint32_t                    *out_flags,
        char                        *out_model_name,
        int                          out_model_name_buflen,
        char                        *out_node_id,
        int                          out_node_id_buflen,
        char                        *out_blob_name,
        int                          out_blob_name_buflen);

/* Copy the rendered RGB bytes for an image-cache entry into out_buf.
   Returns bytes copied, or -1 on error. */
NODUS_API int32_t nodus_weight_image_store_copy_image(
        const NodusWeightImageStore *store,
        int32_t                      index,
        uint8_t                     *out_buf,
        uint64_t                     buf_size);

/* Mark the rendered image associated with state_publish_seq as checkpoint-backed.
   Returns 0 on success, -1 if the image is not present in the cache. */
NODUS_API int nodus_weight_image_store_mark_checkpoint(
        NodusWeightImageStore *store,
        uint64_t               state_publish_seq,
        int32_t                round_id,
        int32_t                cycle);

#ifdef __cplusplus
}
#endif

#endif /* NODUS_LOSS_STORE_H */
