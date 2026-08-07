cmake_minimum_required(VERSION 3.5...4.10)
project(flacoutcpp VERSION 1.2.0)

if(MSVC)
    # force static msvc runtime linkage (/mt for release, /mtd for debug)
    if(POLICY CMP0091)
        cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Without this a plain `cmake -S . -B build` compiles with no -O flag at all,
# which makes the optimizer's hot loops roughly an order of magnitude slower.
# Single-config generators only; multi-config ones pick the type at build time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()

# Opt-in host-specific codegen. Worth ~7% on the exhaustive path, but produces
# binaries that may not run on other machines, so it is off by default.
option(FLACOUT_NATIVE "Optimize for the build host's CPU (not portable)" OFF)

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

if(FLACOUT_NATIVE AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    # Apple Silicon clang wants -mcpu=native; x86 wants -march=native.
    check_cxx_compiler_flag("-mcpu=native" FLACOUT_HAS_MCPU_NATIVE)
    if(FLACOUT_HAS_MCPU_NATIVE)
        target_compile_options(flacout_lib PRIVATE -mcpu=native)
    else()
        check_cxx_compiler_flag("-march=native" FLACOUT_HAS_MARCH_NATIVE)
        if(FLACOUT_HAS_MARCH_NATIVE)
            target_compile_options(flacout_lib PRIVATE -march=native)
        endif()
    endif()
endif()

# CLI executable
add_executable(flacoutcpp src/main.cpp)
target_link_libraries(flacoutcpp flacout_lib)
target_compile_definitions(flacoutcpp PRIVATE FLACOUTCPP_VERSION="${PROJECT_VERSION}")

install(TARGETS flacoutcpp DESTINATION bin)
