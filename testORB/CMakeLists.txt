cmake_minimum_required(VERSION 3.0)
project(testViz)

find_package(OpenCV)
find_package(Egein3)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(runner main.cpp)
    target_link_libraries(runner ${OpenCV_LIBRARIES})

endif()