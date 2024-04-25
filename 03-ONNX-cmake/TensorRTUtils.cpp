#include "TensorRTUtils.h"
using namespace nvinfer1;

/* CUDA调用的检测函数 */
#define CHECK(call) check(call, __LINE__, __FILE__)

/* 检查 cuda runtime API 的运行是否成功 */
bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

class Logger : public ILogger {
    void log(Severity severity, const char* message) noexcept override{
        if(severity != Severity::kINFO)
            std::cout << message << std::endl;
    }
} global_logger;


/** 从文件中读取二进制数据
 * \param[in] path 文件路径
 * \return 读取到的数据
*/
std::string readFile(const std::string& path){
    std::string buffer;
    std::ifstream stream(path.c_str(), std::ios::binary);

    if (stream){
        stream >> std::noskipws;
        std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
        std::cout << "[ReadFile] Read File from " << path << " Successed\n";
    } else{
        std::cout << "[ReadFile] ERROR: Read File from " << path << " Failed !!!!\n";
    }

    return buffer;
}

/** 将二进制数据写入文件
 * \param[in] buffer 二进制数据
 * \param[in] size 数据大小
 * \param[in] path 文件路径
*/
void writeFile(void* buffer, size_t size, const std::string& path) {
    std::ofstream stream(path.c_str(), std::ios::binary);
    if(stream){
        stream.write(static_cast<char*>(buffer), size);
        std::cout << "[WriteFile] Write File to " << path << " Successed\n";
    }
}

/** 创建 cuda 推理引擎
 * \param[in] onnx_model_path onnx模型文件路径
 * \return cuda 推理引擎
*/
ICudaEngine* createCudaEngine(const std::string& onnx_model_path){
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(global_logger)};
    std::unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{builder->createNetworkV2(explicit_batch)};
    std::unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, global_logger)};
    std::unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config{builder->createBuilderConfig()};

    /* 从 onnx 文件中解析模型 */
    if (!parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        std::cout << "[CreateCudaEngine] ERROR: Parse Input Engine Failed !!!\n";
        return nullptr;
    } else {
        std::cout << "[CreateCudaEngine] Parse Input Engine Successed\n";
    }

    /* 设置网络创建信息 */
    config->setMaxWorkspaceSize(1ull << 33); // 显存512MB
    builder->setMaxBatchSize(512 * 512);

    /* 创建优化配置 */
    auto profile = builder->createOptimizationProfile();
    int width = 512, height = 512;
    profile->setDimensions("data_input", OptProfileSelector::kMIN, Dims2(1, 16));
    profile->setDimensions("data_input", OptProfileSelector::kOPT, Dims2(width * height, 16));
    profile->setDimensions("data_input", OptProfileSelector::kMAX, Dims2(width * height * 2, 16));
    profile->setDimensions("data_output", OptProfileSelector::kMIN, Dims2(1, 12));
    profile->setDimensions("data_output", OptProfileSelector::kOPT, Dims2(width * height, 12));
    profile->setDimensions("data_output", OptProfileSelector::kMAX, Dims2(width * height * 2, 12));
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

/** 从文件中获取 cuda 推理引擎
 * \param[in] model_path 文件路径
 * \return cuda 推理引擎
*/
ICudaEngine* getCudaEngine(const std::string& model_path){
    ICudaEngine* engine{nullptr};

    /* 读取 onnx 文件 */
    std::string buffer = readFile(model_path);
    std::cout << "[GetCudaEngine] Read model from path: " << model_path << std::endl;

    /* 尝试 deserialize 引擎*/
    if (buffer.size()) {
        auto runtime = createInferRuntime(global_logger);
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
        if(engine){
            std::cout << "[GetCudaEngine] Deserialize Cuda Engine Successed\n";
        } else {
            std::cout << "[GetCudaEngine] Deserialize Cuda Engine Failed, Try To Build TensorRT Engine\n";
        }
    }

    /* 创建 cuda 推理引擎*/
    if (!engine) {
        engine = createCudaEngine(model_path);
        if (engine) {
            std::cout << "[GetCudaEngine] Build TensorRT Engine Successed\n";
            // 保存为 tensorRT 的文件格式, 供后续使用
            // std::unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
            // writeFile(engine_plan->data(), engine_plan->size(), model_path);
        } else {
            std::cout << "[GetCudaEngine] ERROR: Build TensorRT Engine Failed !!!\n";
        }
    }

    return engine;
}



void TensorRTUtils::initTensorRT(const std::string& model_path) {
    /* 创建推理引擎 */
    engine.reset(getCudaEngine(model_path));
    /* 创建 TensorRT 上下文 */
    context.reset(engine->createExecutionContext());

    /* 输出 binding 信息*/
    std::cout << "[InitTensorRT] Engine has binding num: " << engine->getNbBindings() << std::endl;
    Dims dims_input{ context->getBindingDimensions(0) };
    std::cout << " dim: [";
    for (int k = 0; k < dims_input.nbDims; k++)
        std::cout << dims_input.d[k] << ", ";
    std::cout << "]" << '\n';

    /* 设置 TensorRT 上下文的 binding 信息 */
    context->setBindingDimensions(0, dims_input);
}

void* TensorRTUtils::getSharedDevicePtr(void* shared_handle, uint32_t bytes) {
    if (shared_handle == NULL) return nullptr;

    // 创建 shared memory buffer 的 descriptor
    cudaExternalMemoryHandleDesc external_memory_handle_desc;
    memset(&external_memory_handle_desc, 0, sizeof(external_memory_handle_desc));
    external_memory_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    external_memory_handle_desc.handle.win32.handle = shared_handle;
    external_memory_handle_desc.size = bytes;
    external_memory_handle_desc.flags = cudaExternalMemoryDedicated;

    // 获取 memory 的 handle
    cudaExternalMemory_t external_memory;
    CHECK(cudaImportExternalMemory(&external_memory, &external_memory_handle_desc));

    // 创建 buffer 指针的 descriptor
    cudaExternalMemoryBufferDesc buffer_desc;
    memset(&buffer_desc, 0, sizeof(buffer_desc));
    buffer_desc.size = bytes;

    // 映射 buffer
    void* device_ptr = nullptr;
    CHECK(cudaExternalMemoryGetMappedBuffer(&device_ptr, external_memory, &buffer_desc));
    return device_ptr;
}
