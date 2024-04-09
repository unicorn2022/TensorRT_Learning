#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "cuda.h"
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include "NvOnnxConfig.h"

using namespace nvinfer1;


template <typename T>
struct Destroy {
    void operator()(T* t) const {
        t->destroy();
    }
};

class CudaStream {
public:
    CudaStream() {
        cudaStreamCreate(&mStream_);
    }

    operator cudaStream_t() {
        return mStream_;
    }

    ~CudaStream() {
        cudaStreamDestroy(mStream_);
    }

private:
    cudaStream_t mStream_;
};


/** 初始化 tensorRT 上下文
 * \param[out] engine 推理引擎
 * \param[out] context tensorRT上下文
 * \param[in] model_path 模型路径
*/
void init_tensorRT(
    std::unique_ptr<ICudaEngine, Destroy<ICudaEngine>>& engine, 
    std::unique_ptr<IExecutionContext, Destroy<IExecutionContext>>& context, 
    const std::string& model_path
);

/** 获取 CPU&GPU shared buffer 指针
 * \param[out] shared_handle 
 * \param[out] bytes buffer 大小
 * \return buffer 指针
*/
void* get_shared_device_ptr(void* shared_handle, uint32_t bytes);