cmake_minimum_required(VERSION 2.8.12)
project(ESIM)

# set(CMAKE_CXX_COMPILER             "/usr/bin/g++")
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RELEASE)

add_subdirectory(${CMAKE_SOURCE_DIR}/pybind11)
pybind11_add_module(esim esim.cpp)
# target_link_libraries(esim PUBLIC ${OpenCV_LIBS})