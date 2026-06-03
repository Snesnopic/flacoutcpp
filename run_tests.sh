#!/bin/bash
set -e
cd /Users/giuseppefrancione/flacoutcpp

# BUILD MULTITHREADED FIRST!
cmake --build cmake-build-release

echo "=== MULTITHREADED ==="
time ./cmake-build-release/flacoutcpp test_large1.flac test_large1_mt2.flac
time ./cmake-build-release/flacoutcpp test_large2.flac test_large2_mt2.flac
