/*
 * nodus_weight_image.h -- Weight-to-image renderer for the C object.
 *
 * All weight image rendering lives here.  The backend pushes per-layer /
 * per-unit statistics into the C object; this code turns them into an
 * RGB image.  Three modes:
 *
 *   NODUS_WEIGHT_MODE_PARAMETER_GROUPS   square grid, 1 px per node group
 *   NODUS_WEIGHT_MODE_ARCHITECTURAL_TALL layers as vertical columns
 *   NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE layers as horizontal rows
 *
 * Built as part of the nodus_loss_store shared library.
 */

#ifndef NODUS_WEIGHT_IMAGE_H
#define NODUS_WEIGHT_IMAGE_H

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

/* Render modes. */
#define NODUS_WEIGHT_MODE_PARAMETER_GROUPS    0
#define NODUS_WEIGHT_MODE_ARCHITECTURAL_TALL  1
#define NODUS_WEIGHT_MODE_ARCHITECTURAL_WIDE  2

/* Limits. */
#define NODUS_WEIGHT_MAX_LAYERS  512
#define NODUS_WEIGHT_MAX_UNITS   4096
#define NODUS_WEIGHT_MAX_NODES   1024

/* ------------------------------------------------------------------ */
/*  Data types                                                        */
/* ------------------------------------------------------------------ */

/* Per-unit statistics for one neuron / output channel inside a layer. */
typedef struct NodusWeightUnit {
    float mean;
    float mean_abs;
    float rms;
    float diff_mean_abs;
} NodusWeightUnit;

/* Per-layer (node-group) descriptor. */
typedef struct NodusWeightLayer {
    int32_t          unit_count;
    NodusWeightUnit *units;          /* array of unit_count elements */
} NodusWeightLayer;

/* Per-node-group aggregate (for parameter_groups mode). */
typedef struct NodusWeightNodeGroup {
    float mean;
    float mean_abs;
    float rms;
    float diff_mean_abs;
} NodusWeightNodeGroup;

/* ------------------------------------------------------------------ */
/*  Render API                                                        */
/* ------------------------------------------------------------------ */

/* Render a weight image.
 *
 * mode:          one of NODUS_WEIGHT_MODE_*
 *
 * For ARCHITECTURAL_TALL / ARCHITECTURAL_WIDE:
 *   layers:      array of num_layers NodusWeightLayer descriptors
 *   num_layers:  number of layers
 *
 * For PARAMETER_GROUPS:
 *   nodes:       array of num_nodes NodusWeightNodeGroup descriptors
 *   num_nodes:   number of node groups
 *   (layers/num_layers are ignored)
 *
 * target_w, target_h:  desired output image dimensions (minimum 8)
 *
 * out_rgb:       caller-allocated buffer, must be >= target_w * target_h * 3
 *                The image is written as packed RGB, row-major.
 * out_w, out_h:  actual image dimensions written (may exceed target if the
 *                raw grid is larger than target).
 *
 * Returns 0 on success, -1 on error.
 */
NODUS_API int nodus_weight_image_render(
        int                         mode,
        const NodusWeightLayer     *layers,
        int32_t                     num_layers,
        const NodusWeightNodeGroup *nodes,
        int32_t                     num_nodes,
        int32_t                     target_w,
        int32_t                     target_h,
        uint8_t                    *out_rgb,
        int32_t                    *out_w,
        int32_t                    *out_h);

/* Compute the exact rendered image size without rendering pixels.
 *
 * Inputs mirror nodus_weight_image_render(), minus the output buffer.
 * Returns 0 on success, -1 on error.
 */
NODUS_API int nodus_weight_image_measure(
        int                         mode,
        const NodusWeightLayer     *layers,
        int32_t                     num_layers,
        const NodusWeightNodeGroup *nodes,
        int32_t                     num_nodes,
        int32_t                     target_w,
        int32_t                     target_h,
        int32_t                    *out_w,
        int32_t                    *out_h);

/* ------------------------------------------------------------------ */
/*  State-dict intake: dump raw tensors, C does everything            */
/* ------------------------------------------------------------------ */

/* Accept a raw state_dict (parallel arrays of names, float data, element
 * counts, and leading-dimension sizes) and produce a weight image.
 *
 * param_names:   array of num_params NUL-terminated UTF-8 parameter keys
 *                (e.g. "block.weight", "block.bias", "head.weight").
 * param_data:    array of num_params float pointers, each pointing to the
 *                contiguous float32 tensor data.
 * param_numel:   number of float elements per parameter.
 * param_shape0:  size of dimension 0 of each parameter tensor (for unit
 *                counting; scalars should pass 1).
 * num_params:    total parameter count.
 *
 * ref_data:      optional reference state (same layout) for diff
 *                computation.  NULL to skip diff.
 * ref_numel:     element counts for ref_data (must match param layout).
 *
 * mode:          NODUS_WEIGHT_MODE_*
 * target_w, target_h: desired output dimensions.
 *
 * out_rgb:       set to a heap-allocated RGB buffer on success.  Caller
 *                must free with nodus_weight_image_free().
 * out_w, out_h:  actual image dimensions.
 *
 * Returns 0 on success, -1 on error.
 */
NODUS_API int nodus_weight_image_from_state_dict(
        const char *const  *param_names,
        const float *const *param_data,
        const int32_t      *param_numel,
        const int32_t      *param_shape0,
        int32_t             num_params,
        const float *const *ref_data,
        const int32_t      *ref_numel,
        int                 mode,
        int32_t             target_w,
        int32_t             target_h,
        uint8_t           **out_rgb,
        int32_t            *out_w,
        int32_t            *out_h);

/* State-dict variant of nodus_weight_image_measure(). */
NODUS_API int nodus_weight_image_measure_from_state_dict(
        const char *const  *param_names,
        const float *const *param_data,
        const int32_t      *param_numel,
        const int32_t      *param_shape0,
        int32_t             num_params,
        int                 mode,
        int32_t             target_w,
        int32_t             target_h,
        int32_t            *out_w,
        int32_t            *out_h);

/* Free a buffer returned by nodus_weight_image_from_state_dict. */
NODUS_API void nodus_weight_image_free(uint8_t *rgb);

#ifdef __cplusplus
}
#endif

#endif /* NODUS_WEIGHT_IMAGE_H */
