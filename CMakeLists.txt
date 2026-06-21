cmake_minimum_required(VERSION 3.5...4.10)
project(flacoutcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimisation for Release builds.
# -march=native  → use all CPU features (NEON on Apple Silicon, AVX2 on x86)
# -ffast-math    → enables vectorisation of FP loops (autocorr, residuals)
# -O3            → already set by CMake for Release, but make it explicit
if(NOT MSVC)
    add_compile_options(
        "$<$<CONFIG:Release>:-O3>"
        "$<$<CONFIG:Release>:-march=native>"
        "$<$<CONFIG:Release>:-ffast-math>"
    )
endif()


set(WITH_OGG OFF CACHE BOOL "Disable Ogg support in vendored FLAC" FORCE)
set(INSTALL_MANPAGES OFF CACHE BOOL "Don't build FLAC manpages" FORCE)
add_subdirectory(third_party/libflac EXCLUDE_FROM_ALL)

include_directories(
    include
    third_party/libflac/include
)

# Core library (optimizer + processor + frame writer + bitwriter + library shim)
add_library(flacout_lib
    src/optimizer.cpp
    src/processor.cpp
    src/bitwriter.cpp
    src/frame_writer.cpp
    src/flacoutcpp.cpp
)
target_link_libraries(flacout_lib FLAC)

# CLI executable
add_executable(flacoutcpp src/main.cpp)
target_link_libraries(flacoutcpp flacout_lib)

# ---------------------------------------------------------------------------
# Warning flags  (applied only to our targets, never to third_party/libflac)
# ---------------------------------------------------------------------------
# Clang: -Weverything with targeted suppressions for intentional choices.
# GCC:   -Wall -Wextra -Wpedantic (GCC has no -Weverything equivalent).
# MSVC:  /W4 (future-proofing).
# ---------------------------------------------------------------------------
set(FLACOUT_CLANG_SUPPRESSIONS
    # Language-version compat: we target C++17, not C++98.
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-pre-c++17-compat
    # Style: trailing return types are a valid C++11 syntax choice.
    -Wno-c++11-compat
    # Struct padding is intentional in some hot-path structs.
    -Wno-padded
    # Static objects with non-trivial destructors: acceptable in this project.
    -Wno-exit-time-destructors
    -Wno-global-constructors
    # Virtual tables: we have no polymorphic hierarchy.
    -Wno-weak-vtables
    # switch/default: covered-switch-default is too noisy on enums with all cases listed.
    -Wno-switch-default
    -Wno-covered-switch-default
    # Unsafe buffer access: too noisy when working with raw PCM arrays.
    -Wno-unsafe-buffer-usage
    # double-promotion: we intentionally promote float→double in LPC.
    -Wno-double-promotion
    # These conflict with POSIX / system headers on some platforms.
    -Wno-reserved-identifier
    -Wno-reserved-macro-identifier
    # Disabled: suggest explicit braces is purely cosmetic, not a bug class.
    -Wno-misleading-indentation
)

foreach(tgt IN ITEMS flacout_lib flacoutcpp)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${tgt} PRIVATE
            -Weverything
            ${FLACOUT_CLANG_SUPPRESSIONS}
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${tgt} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wshadow -Wcast-align -Wdouble-promotion
            -Wformat=2 -Wundef
        )
    elseif(MSVC)
        target_compile_options(${tgt} PRIVATE /W4)
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Optional: clang-tidy integration
# Enable by passing -DENABLE_CLANG_TIDY=ON to cmake.
# A .clang-tidy file in the project root controls which checks are active.
# ---------------------------------------------------------------------------
option(ENABLE_CLANG_TIDY "Run clang-tidy on every translation unit" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else()
        message(WARNING "ENABLE_CLANG_TIDY=ON but clang-tidy not found in PATH")
    endif()
endif()

install(TARGETS flacoutcpp DESTINATION bin)
