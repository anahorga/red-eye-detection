cmake_minimum_required(VERSION 3.25)
project(RedEyeDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(RedEyeDetection main.cpp)

target_link_libraries(RedEyeDetection ${OpenCV_LIBS})