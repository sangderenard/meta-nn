/*
 * nodus_onnx.cpp -- ONNX Runtime inference (+ optional training) bridge.
 *
 * Consumes .onnx models produced by torch.onnx.export and exposes a
 * C-linkage API (nodus_onnx.h) suitable for Python ctypes, Emscripten
 * WebAssembly, or direct native callers.
 *
 * Build requirements:
 *   - ONNX Runtime C API headers  (onnxruntime_c_api.h)
 *   - Optionally: onnxruntime_training_c_api.h → compile with
 *     -DNODUS_ONNX_TRAINING=1 to enable the train_step / save APIs.
 *
 * The implementation is deliberately minimal: one input, one output,
 * float32 only.  Extend as more model topologies are exported.
 */

#include "nodus_onnx.h"
#include <onnxruntime_c_api.h>

#ifdef NODUS_ONNX_TRAINING
#  include <onnxruntime_training_c_api.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Thread-local error buffer                                         */
/* ------------------------------------------------------------------ */

static thread_local std::string g_last_error;

static void _set_error(const char *msg) {
    g_last_error = msg ? msg : "";
}

static void _set_ort_error(const OrtApi *api, OrtStatus *status) {
    if (status) {
        _set_error(api->GetErrorMessage(status));
        api->ReleaseStatus(status);
    }
}

/* ------------------------------------------------------------------ */
/*  Session structure                                                 */
/* ------------------------------------------------------------------ */

struct NodusOnnxSession {
    const OrtApi       *api          = nullptr;
    OrtEnv             *env          = nullptr;
    OrtSessionOptions  *session_opts = nullptr;
    OrtSession         *session      = nullptr;
    OrtMemoryInfo      *mem_info     = nullptr;
    OrtAllocator       *allocator    = nullptr;

    /* Cached input/output names (owned strings). */
    std::string input_name;
    std::string output_name;

#ifdef NODUS_ONNX_TRAINING
    const OrtTrainingApi *train_api  = nullptr;
    OrtTrainingSession   *train_sess = nullptr;
#endif
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

#define ORT_CHECK(expr)                             \
    do {                                            \
        OrtStatus *_s = (expr);                     \
        if (_s) { _set_ort_error(api, _s); return nullptr; } \
    } while (0)

#define ORT_CHECK_INT(expr, retval)                 \
    do {                                            \
        OrtStatus *_s = (expr);                     \
        if (_s) { _set_ort_error(s->api, _s); return retval; } \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

extern "C" NODUS_API NodusOnnxSession *nodus_onnx_create(const char *model_path) {
    if (!model_path || !*model_path) {
        _set_error("model_path is null or empty");
        return nullptr;
    }

    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        _set_error("failed to obtain ORT API");
        return nullptr;
    }

    NodusOnnxSession *s = new (std::nothrow) NodusOnnxSession();
    if (!s) { _set_error("allocation failed"); return nullptr; }
    s->api = api;

    ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nodus_onnx", &s->env));
    ORT_CHECK(api->CreateSessionOptions(&s->session_opts));
    ORT_CHECK(api->SetIntraOpNumThreads(s->session_opts, 1));
    ORT_CHECK(api->SetSessionGraphOptimizationLevel(s->session_opts, ORT_ENABLE_EXTENDED));

#ifdef _WIN32
    /* Windows: convert UTF-8 path to wide string for CreateSession. */
    int wlen = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath.data(), wlen);
    ORT_CHECK(api->CreateSession(s->env, wpath.data(), s->session_opts, &s->session));
#else
    ORT_CHECK(api->CreateSession(s->env, model_path, s->session_opts, &s->session));
#endif

    ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->mem_info));
    ORT_CHECK(api->GetAllocatorWithDefaultOptions(&s->allocator));

    /* Cache first input/output name. */
    {
        char *name = nullptr;
        ORT_CHECK(api->SessionGetInputName(s->session, 0, s->allocator, &name));
        s->input_name = name;
        api->AllocatorFree(s->allocator, name);
    }
    {
        char *name = nullptr;
        ORT_CHECK(api->SessionGetOutputName(s->session, 0, s->allocator, &name));
        s->output_name = name;
        api->AllocatorFree(s->allocator, name);
    }

#ifdef NODUS_ONNX_TRAINING
    s->train_api = OrtGetApiBase()->GetTrainingApi(ORT_API_VERSION);
    /* Training session creation is deferred to first train_step call
       because it requires additional artifacts (checkpoint, optimizer). */
#endif

    g_last_error.clear();
    return s;
}

extern "C" NODUS_API void nodus_onnx_destroy(NodusOnnxSession *s) {
    if (!s) return;
    const OrtApi *api = s->api;
#ifdef NODUS_ONNX_TRAINING
    if (s->train_sess && s->train_api)
        s->train_api->ReleaseTrainingSession(s->train_sess);
#endif
    if (s->session)      api->ReleaseSession(s->session);
    if (s->session_opts) api->ReleaseSessionOptions(s->session_opts);
    if (s->mem_info)     api->ReleaseMemoryInfo(s->mem_info);
    if (s->env)          api->ReleaseEnv(s->env);
    delete s;
}

/* ------------------------------------------------------------------ */
/*  Inference                                                         */
/* ------------------------------------------------------------------ */

extern "C" NODUS_API int64_t nodus_onnx_infer(
    NodusOnnxSession *s,
    const float      *input_data,
    const int64_t    *input_dims,
    int32_t           input_ndims,
    float            *output_data,
    int64_t           output_capacity)
{
    if (!s || !input_data || !input_dims || input_ndims < 1 || !output_data) {
        _set_error("invalid arguments");
        return -1;
    }
    const OrtApi *api = s->api;

    /* Compute total element count for the input tensor. */
    int64_t input_numel = 1;
    for (int32_t i = 0; i < input_ndims; ++i) input_numel *= input_dims[i];

    /* Create input tensor. */
    OrtValue *input_tensor = nullptr;
    ORT_CHECK_INT(api->CreateTensorWithDataAsOrtValue(
        s->mem_info,
        const_cast<float *>(input_data), input_numel * sizeof(float),
        input_dims, input_ndims,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor), -1);

    /* Run. */
    const char *in_names[]  = { s->input_name.c_str()  };
    const char *out_names[] = { s->output_name.c_str() };
    OrtValue *output_tensor = nullptr;
    OrtStatus *run_status = api->Run(
        s->session, nullptr,
        in_names, (const OrtValue *const *)&input_tensor, 1,
        out_names, 1, &output_tensor);
    api->ReleaseValue(input_tensor);
    if (run_status) {
        _set_ort_error(api, run_status);
        return -1;
    }

    /* Copy output data. */
    float *out_ptr = nullptr;
    ORT_CHECK_INT(api->GetTensorMutableData(output_tensor, (void **)&out_ptr), -1);

    OrtTensorTypeAndShapeInfo *info = nullptr;
    ORT_CHECK_INT(api->GetTensorTypeAndShape(output_tensor, &info), -1);
    size_t out_numel = 0;
    api->GetTensorShapeElementCount(info, &out_numel);
    api->ReleaseTensorTypeAndShapeInfo(info);

    int64_t to_copy = (int64_t)out_numel;
    if (to_copy > output_capacity) to_copy = output_capacity;
    std::memcpy(output_data, out_ptr, to_copy * sizeof(float));

    api->ReleaseValue(output_tensor);
    g_last_error.clear();
    return to_copy;
}

/* ------------------------------------------------------------------ */
/*  Training (stub when NODUS_ONNX_TRAINING is not set)               */
/* ------------------------------------------------------------------ */

#ifdef NODUS_ONNX_TRAINING

extern "C" NODUS_API float nodus_onnx_train_step(
    NodusOnnxSession *s,
    const float      *input_data,
    const int64_t    *input_dims,
    int32_t           input_ndims,
    const float      *target_data,
    const int64_t    *target_dims,
    int32_t           target_ndims,
    float             learning_rate)
{
    if (!s || !s->train_api) {
        _set_error("training API not available");
        return NAN;
    }

    /*
     * ONNX Runtime Training requires:
     *   1. A training model artifact (.onnx with loss + gradients)
     *   2. An optimizer model artifact (.onnx)
     *   3. A checkpoint directory
     *
     * These are produced by onnxruntime.training.artifacts.generate_artifacts().
     * For now this is a structural placeholder — the real flow will be:
     *   Python: torch model → ONNX → generate training artifacts
     *   C++:    load artifacts here → train_step → save checkpoint
     */

    /* TODO: implement once training artifact generation is wired up.
       The inference path is complete and usable now. */
    _set_error("training not yet wired — awaiting artifact generation pipeline");
    return NAN;
}

extern "C" NODUS_API int nodus_onnx_save(NodusOnnxSession *s,
                                         const char *output_path) {
    if (!s || !s->train_api) {
        _set_error("training API not available");
        return -1;
    }
    /* TODO: save checkpoint via OrtTrainingApi once train_step is implemented. */
    _set_error("save not yet wired — awaiting training implementation");
    return -1;
}

#else /* !NODUS_ONNX_TRAINING */

extern "C" NODUS_API float nodus_onnx_train_step(
    NodusOnnxSession *, const float *, const int64_t *, int32_t,
    const float *, const int64_t *, int32_t, float)
{
    _set_error("training support not compiled (NODUS_ONNX_TRAINING=0)");
    return NAN;
}

extern "C" NODUS_API int nodus_onnx_save(NodusOnnxSession *, const char *) {
    _set_error("training support not compiled (NODUS_ONNX_TRAINING=0)");
    return -1;
}

#endif /* NODUS_ONNX_TRAINING */

/* ------------------------------------------------------------------ */
/*  Error reporting                                                   */
/* ------------------------------------------------------------------ */

extern "C" NODUS_API const char *nodus_onnx_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}
