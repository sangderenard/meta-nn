/*
 * nodus_onnx.h -- ONNX Runtime inference (and optional training) for nodus.
 *
 * Loads an .onnx model exported by torch.onnx.export (via the nodus HTTP
 * server's /api/onnx/export endpoint) and provides C-linkage functions for
 * inference and, when built with NODUS_ONNX_TRAINING=1, on-device training.
 *
 * The API is designed to be consumed from Python (ctypes), JavaScript
 * (Emscripten / WebAssembly), or any native caller.
 */

#ifndef NODUS_ONNX_H
#define NODUS_ONNX_H

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
/*  Opaque handle                                                     */
/* ------------------------------------------------------------------ */

typedef struct NodusOnnxSession NodusOnnxSession;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/*  Create an inference session from an .onnx file on disk.
 *  Returns NULL on failure; call nodus_onnx_last_error() for details. */
NODUS_API NodusOnnxSession *nodus_onnx_create(const char *model_path);

/*  Destroy a session and free all resources. */
NODUS_API void nodus_onnx_destroy(NodusOnnxSession *session);

/* ------------------------------------------------------------------ */
/*  Inference                                                         */
/* ------------------------------------------------------------------ */

/*  Run inference on a single float tensor.
 *
 *  input_data:      pointer to contiguous float32 input
 *  input_dims:      shape array, e.g. {1, 3, 64, 64}
 *  input_ndims:     number of dimensions (length of input_dims)
 *  output_data:     caller-allocated buffer for output float32s
 *  output_capacity: number of floats that fit in output_data
 *
 *  Returns the number of floats written to output_data, or -1 on error.
 */
NODUS_API int64_t nodus_onnx_infer(
    NodusOnnxSession *session,
    const float      *input_data,
    const int64_t    *input_dims,
    int32_t           input_ndims,
    float            *output_data,
    int64_t           output_capacity);

/* ------------------------------------------------------------------ */
/*  Training  (only available when built with NODUS_ONNX_TRAINING=1)  */
/* ------------------------------------------------------------------ */

/*  Run a single training step: forward + backward + optimizer step.
 *
 *  input_data / input_dims / input_ndims: model input (same as infer).
 *  target_data / target_dims / target_ndims: ground-truth targets.
 *  learning_rate: SGD step size.
 *
 *  Returns the scalar loss, or NaN on error.
 */
NODUS_API float nodus_onnx_train_step(
    NodusOnnxSession *session,
    const float      *input_data,
    const int64_t    *input_dims,
    int32_t           input_ndims,
    const float      *target_data,
    const int64_t    *target_dims,
    int32_t           target_ndims,
    float             learning_rate);

/*  Save the current (possibly trained) weights back to an .onnx file. */
NODUS_API int nodus_onnx_save(NodusOnnxSession *session,
                              const char *output_path);

/* ------------------------------------------------------------------ */
/*  Error reporting                                                   */
/* ------------------------------------------------------------------ */

/*  Thread-local error string from the last failed call, or NULL. */
NODUS_API const char *nodus_onnx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NODUS_ONNX_H */
