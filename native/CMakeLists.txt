cmake_minimum_required(VERSION 3.15)
project(nodus_native C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow isolated build outputs (e.g. debug builds while release DLL is in use).
set(NODUS_OUTPUT_DIR "" CACHE PATH "Optional output directory for nodus_loss_store binaries")

# ---------------------------------------------------------------------------
#  nodus_loss_store  (C shared library — existing)
# ---------------------------------------------------------------------------

add_library(nodus_loss_store SHARED nodus_loss_store.c nodus_weight_image.c)

target_include_directories(nodus_loss_store PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Export macros
target_compile_definitions(nodus_loss_store PRIVATE NODUS_BUILDING_DLL)

# Platform libs
if(UNIX)
    target_link_libraries(nodus_loss_store PRIVATE pthread)
    set_target_properties(nodus_loss_store PROPERTIES
        C_VISIBILITY_PRESET hidden
    )
endif()

# Output defaults to pipeline/ for compatibility, but can be overridden.
if(NODUS_OUTPUT_DIR)
    set(_NODUS_BIN_DIR "${NODUS_OUTPUT_DIR}")
else()
    set(_NODUS_BIN_DIR "${CMAKE_SOURCE_DIR}/../pipeline")
endif()

set_target_properties(nodus_loss_store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_NODUS_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${_NODUS_BIN_DIR}"
)

# Also place into per-config dirs for multi-config generators (VS)
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${cfg} CFG_UPPER)
    set_target_properties(nodus_loss_store PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} "${_NODUS_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} "${_NODUS_BIN_DIR}"
    )
endforeach()

# ---------------------------------------------------------------------------
#  nodus_onnx  (C++ shared library — ONNX Runtime inference + training)
# ---------------------------------------------------------------------------
#
# Set ONNXRUNTIME_ROOT to the ONNX Runtime SDK directory, e.g.:
#   cmake -DONNXRUNTIME_ROOT=C:/onnxruntime-win-x64-1.17.0 ..
#
# When the SDK is not available the target is silently skipped so the rest
# of the build is not affected.
# ---------------------------------------------------------------------------

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime SDK (include/ + lib/)")

if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/include")
    message(STATUS "ONNX Runtime SDK found: ${ONNXRUNTIME_ROOT}")

    add_library(nodus_onnx SHARED nodus_onnx.cpp)

    target_include_directories(nodus_onnx PRIVATE "${ONNXRUNTIME_ROOT}/include")
    target_compile_definitions(nodus_onnx PRIVATE NODUS_BUILDING_DLL)

    # Link the ONNX Runtime shared library
    if(WIN32)
        target_link_libraries(nodus_onnx PRIVATE
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    else()
        target_link_libraries(nodus_onnx PRIVATE
            "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()

    # Check for training headers (optional — inference is always available)
    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_training_c_api.h")
        target_compile_definitions(nodus_onnx PRIVATE NODUS_ONNX_TRAINING=1)
        message(STATUS "  ONNX Runtime Training API headers detected — training enabled")
    else()
        message(STATUS "  ONNX Runtime Training API headers NOT found — inference only")
    endif()

    set_target_properties(nodus_onnx PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_NODUS_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${_NODUS_BIN_DIR}"
    )
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        string(TOUPPER ${cfg} CFG_UPPER)
        set_target_properties(nodus_onnx PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} "${_NODUS_BIN_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} "${_NODUS_BIN_DIR}"
        )
    endforeach()
else()
    message(STATUS "ONNXRUNTIME_ROOT not set or invalid — skipping nodus_onnx target. "
                   "Pass -DONNXRUNTIME_ROOT=<sdk_dir> to enable.")
endif()
