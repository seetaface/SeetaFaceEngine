cmake_minimum_required(VERSION 2.8.4)

project(seeta_fa_lib)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  OFF)

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

# Install target
INSTALL(TARGETS ${PROJECT_NAME}
		RUNTIME DESTINATION bin
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")
# Install head files
INSTALL(DIRECTORY include DESTINATION ".")
# Install model files
INSTALL(DIRECTORY model DESTINATION ".")

if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found. Test will not be built.")
    endif()

	find_package(SeetaFaceEngine)
	if(NOT SEETAFACEENGINE_FOUND)
		message(FATAL_ERROR "Please set BUILD_EXAMPLES to OFF and isntall SeetaFaceEngine")
	endif()

	include_directories(${OpenCV_INCLUDE_DIRS} ${SEETAFACEENGINE_INCLUDE_DIRS})
	add_executable(fa_test src/test/face_alignment_test.cpp)
	target_link_libraries(fa_test ${OpenCV_LIBS} ${SEETAFACEENGINE_LIBRARIES})

	# Install target
	INSTALL(TARGETS fa_test RUNTIME DESTINATION bin)
	# Install example data
	INSTALL(DIRECTORY data DESTINATION ".")

endif()
