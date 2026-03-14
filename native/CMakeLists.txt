cmake_minimum_required(VERSION 3.15)
project(nodus_loss_store C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Shared library
add_library(nodus_loss_store SHARED nodus_loss_store.c)

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

# Output directly into the project's pipeline/ directory so Python can find it
set_target_properties(nodus_loss_store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../pipeline"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../pipeline"
)

# Also place into per-config dirs for multi-config generators (VS)
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${cfg} CFG_UPPER)
    set_target_properties(nodus_loss_store PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_SOURCE_DIR}/../pipeline"
        RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_SOURCE_DIR}/../pipeline"
    )
endforeach()
