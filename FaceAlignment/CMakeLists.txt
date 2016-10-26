cmake_minimum_required(VERSION 2.8.4)

project(seeta_fa_lib)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)

# Use C++11
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "C++11 support has been enabled by default.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

include_directories(include)

set(src_files 
    src/cfan.cpp
    src/face_alignment.cpp
    src/sift.cpp
    )

add_library(seeta_fa_lib SHARED ${src_files})
set(fa_required_libs seeta_fa_lib)

if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS} build)
	link_directories(build)
        list(APPEND fa_required_libs ${OpenCV_LIBS} seeta_facedet_lib)
        add_executable(fa_test src/test/face_alignment_test.cpp)
        target_link_libraries(fa_test ${fa_required_libs})
    endif()
endif()
