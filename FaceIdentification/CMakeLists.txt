cmake_minimum_required (VERSION 2.6)

project (viplnet)

# Build options
option(BUILD_EXAMPLES "Set to ON to build examples" OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGTEST_USE_OWN_TR1_TUPLE=1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGTEST_USE_OWN_TR1_TUPLE=1")
endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

include_directories(include)

set(src_files 
    src/blob.cpp
    src/bn_net.cpp
	src/bias_adder_net.cpp
	src/common_net.cpp
	src/conv_net.cpp
	src/eltwise_net.cpp
	src/inner_product_net.cpp
	src/log.cpp
	src/math_functions.cpp
	src/max_pooling_net.cpp
	src/net.cpp
	src/pad_net.cpp
	src/spatial_transform_net.cpp
	src/tform_maker_net.cpp
    )

set(tools_files 
	tools/aligner.cpp
	tools/face_identification.cpp
	)

#aux_source_directory(./src src_files)
#aux_source_directory(./tools tools_files)

add_library(viplnet SHARED ${src_files} ${tools_files})
set_target_properties(viplnet PROPERTIES 
  VERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR} 
  SOVERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR}) 

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
	list(APPEND viplnet_required_libs ${OpenCV_LIBS} ${SEETAFACEENGINE_LIBRARIES})

	add_executable(test_face_recognizer src/test/test_face_recognizer.cpp)
    target_link_libraries(test_face_recognizer ${viplnet_required_libs})

	add_executable(test_face_verification src/test/test_face_verification.cpp)
    target_link_libraries(test_face_verification ${viplnet_required_libs})

	# Install target
	INSTALL(TARGETS test_face_recognizer test_face_verification RUNTIME DESTINATION bin)
	# Install example data
	INSTALL(DIRECTORY data DESTINATION ".")
endif()