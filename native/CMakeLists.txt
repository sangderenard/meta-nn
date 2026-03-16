cmake_minimum_required(VERSION 3.15)
project(nodus_loss_store C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Allow isolated build outputs (e.g. debug builds while release DLL is in use).
set(NODUS_OUTPUT_DIR "" CACHE PATH "Optional output directory for nodus_loss_store binaries")

# Shared library
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
