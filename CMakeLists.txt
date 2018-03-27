cmake_minimum_required(VERSION 3.9)
project(videoStabilizator)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES videoStabilizator/main.cpp videoStabilizator/dump.cpp)
add_executable(videoStabilizator ${SOURCE_FILES})
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( videoStabilizator ${OpenCV_LIBS} )
