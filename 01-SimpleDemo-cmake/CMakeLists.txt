cmake_minimum_required(VERSION 3.25)
project(tensorRT_test)
set(CMAKE_CXX_STANDARD 11)
 
set(TENSORRT_ROOT "D:/Program/TensorRT/TensorRT-8.5.3.1")
 
# 包含CUDA的头文件路径
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
 
# 包含TensorRT的头文件路径
include_directories(${TENSORRT_ROOT}/include)

# 添加可执行文件
add_executable(tensorRT_test main.cpp cookbookHelper.cuh)
 
# 链接TensorRT的库文件
link_directories(${TENSORRT_ROOT}/lib)
target_link_libraries(tensorRT_test ${TENSORRT_ROOT}/lib/*.lib)
 
# 链接CUDA的库文件
target_link_libraries(tensorRT_test ${CUDA_LIBRARIES})