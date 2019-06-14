cmake_minimum_required(VERSION 3.0)
project(testViz)

find_package(OpenCV)
include_directories("../../opencv-3.4.5/install/include")
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(runner main.cpp)
    target_link_libraries(runner ${OpenCV_LIBRARIES})

endif()