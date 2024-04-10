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


class TensorRTUtils{
public:
    /**
     * \param[in] model_path 模型路径
     * \param[in] input_size 输入通道数
     * \param[in] output_size 输出通道数
    */
    TensorRTUtils(const std::string& model_path, int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        initTensorRT(model_path);
    }

    /** 获取 CPU&GPU shared buffer 指针
     * \param[out] shared_handle 
     * \param[out] bytes buffer 大小
     * \return buffer 指针
    */
    void* getSharedDevicePtr(void* shared_handle, uint32_t bytes);

    ICudaEngine* getTensorRTEngine() { return engine.get(); }
    IExecutionContext* getTensorRTContext() { return context.get(); } 

private:
    std::unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    std::unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr}; 
    const int input_size;
    const int output_size;

private:
    /** 初始化 tensorRT 上下文
     * \param[in] model_path 模型路径
    */
    void initTensorRT(const std::string& model_path);
};

