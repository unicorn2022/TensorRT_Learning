#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "Falcor.h"
#include "cuda.h"
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include "NvOnnxConfig.h"

template <typename T>
struct Destroy {
    void operator()(T* t) const {
        t->destroy();
    }
};


class TensorRTUtils{
public:
    TensorRTUtils(){}

    /**
     * \param[in] model_path 模型路径
    */
    TensorRTUtils(const std::string& model_path){
        initTensorRT(model_path);
    }

    /** 获取 CPU&GPU shared buffer 指针
     * \param[out] shared_handle
     * \param[out] bytes buffer 大小
     * \return buffer 指针
    */
    void* getSharedDevicePtr(void* shared_handle, uint32_t bytes);

    nvinfer1::ICudaEngine* getTensorRTEngine() { return engine.get(); }
    nvinfer1::IExecutionContext* getTensorRTContext() { return context.get(); }

private:
    std::unique_ptr<nvinfer1::ICudaEngine, Destroy<nvinfer1::ICudaEngine>> engine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext, Destroy<nvinfer1::IExecutionContext>> context{nullptr};

private:
    /** 初始化 tensorRT 上下文
     * \param[in] model_path 模型路径
    */
    void initTensorRT(const std::string& model_path);
};

