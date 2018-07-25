cmake_minimum_required (VERSION 2.6)

project (viplnet)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)

set (VIPLNET_VERSION_MAJOR 4)
set (VIPLNET_VERSION_MINOR 5)

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
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")
# Install head files
INSTALL(DIRECTORY include DESTINATION ".")

if (BUILD_EXAMPLES)
    set(viplnet_required_libs viplnet)
	
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS} ../FaceDetection/include ../FaceAlignment/include)
	    
        list(APPEND viplnet_required_libs ${OpenCV_LIBS} viplnet seeta_fa_lib seeta_facedet_lib)
        
		add_executable(test_face_recognizer src/test/test_face_recognizer.cpp)
        target_link_libraries(test_face_recognizer ${viplnet_required_libs})
		
		add_executable(test_face_verification src/test/test_face_verification.cpp)
        target_link_libraries(test_face_verification ${viplnet_required_libs})
    endif()
endif()