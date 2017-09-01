aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)
include_directories(${seeta_facedet_lib_INCLUDE_DIRS} ${seeta_fa_lib_INCLUDE_DIRS})

list(APPEND seeta_fi_lib_required_libs ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)
enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${seeta_fi_lib_required_libs})
endforeach ()

