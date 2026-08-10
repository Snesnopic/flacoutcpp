cmake_minimum_required(VERSION 3.5...4.10)
project(flacoutcpp VERSION 1.3.0)

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

# Skip vendoring our own copy if a FLAC target already exists — e.g. when
# flacoutcpp is add_subdirectory'd into a parent project that already
# provides one (its own libflac submodule), so the two don't collide.
if(NOT TARGET FLAC)
    set(WITH_OGG OFF CACHE BOOL "Disable Ogg support in vendored FLAC" FORCE)
    set(INSTALL_MANPAGES OFF CACHE BOOL "Don't build FLAC manpages" FORCE)
    add_subdirectory(third_party/libflac EXCLUDE_FROM_ALL)
endif()

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

# Pin floating-point contraction off.
#
# The optimizer picks LPC coefficients from a double-precision autocorrelation
# and then quantizes them to 8-15 bit integers. A last-bit difference in that
# sum almost always quantizes to the same integer, but occasionally lands one
# step over — different residuals, different cost, a different candidate wins.
# So whether the compiler fuses `a += x * y` into a single-rounding FMA is
# visible in the encoded output.
#
# Left to the compiler, that makes the bitstream a function of the toolchain:
# clang at -O3 and the same clang with -ffp-contract=off disagree on 7 of the 14
# bench/check.sh cases. For a tool whose whole purpose is deterministic optimal
# encoding, that is a defect. Pinning it also lets the autocorrelation be
# restructured for speed without perturbing results, which is worth 1.8x on
# ranked search.
#
# Costs nothing measurable — the FP work is a rounding error next to the integer
# residual and Rice loops.
if(MSVC)
    # MSVC does not contract under /fp:precise, which is also its default. Set
    # it explicitly so a project-wide /fp:fast cannot silently re-enable it.
    target_compile_options(flacout_lib PRIVATE /fp:precise)
else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-ffp-contract=off" FLACOUT_HAS_FP_CONTRACT_OFF)
    if(FLACOUT_HAS_FP_CONTRACT_OFF)
        target_compile_options(flacout_lib PRIVATE -ffp-contract=off)
    else()
        message(WARNING
            "Compiler does not accept -ffp-contract=off; encoded output may "
            "differ from other builds. See the comment in CMakeLists.txt.")
    endif()
endif()

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
