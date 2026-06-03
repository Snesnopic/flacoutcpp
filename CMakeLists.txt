cmake_minimum_required(VERSION 3.5...4.10)
project(flacoutcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_OGG OFF CACHE BOOL "Disable Ogg support in vendored FLAC" FORCE)
add_subdirectory(third_party/libflac EXCLUDE_FROM_ALL)

# Include directory di libFLAC
include_directories(
        include
        third_party/libflac/include
)

# Library
add_library(flacout_lib src/optimizer.cpp src/processor.cpp)
target_link_libraries(flacout_lib FLAC)

# CLI
add_executable(flacoutcpp src/main.cpp)
target_link_libraries(flacoutcpp flacout_lib)
