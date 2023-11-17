cmake_minimum_required(VERSION 3.0.0)
project(layout_detection)

set(ONNXRUNTIME_DIR E:\\onnxruntime-win-x64-1.15.1)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

aux_source_directory(./src SOURCE_SRC)
include_directories("include/")

add_executable(layout_detection ${SOURCE_SRC})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(layout_detection PRIVATE "${ONNXRUNTIME_DIR}/include")

target_compile_features(layout_detection PRIVATE cxx_std_17)
target_link_libraries(layout_detection ${OpenCV_LIBS})

if (WIN32)
    target_link_libraries(layout_detection "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    target_link_libraries(layout_detection "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

