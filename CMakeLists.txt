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

install(TARGETS flacoutcpp DESTINATION bin)
