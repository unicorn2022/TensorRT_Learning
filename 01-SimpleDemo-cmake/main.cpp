#include "cookbookHelper.cuh"

using namespace nvinfer1;

// 文件路径
const std::string trtFile {"./model.plan"};
// 创建 Logger, 包含的日志等级为: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
static Logger gLogger(ILogger::Severity::kERROR);

void run() {
    ICudaEngine *engine = nullptr;

    /* 载入 .plan 序列化网络模型 */ 
    if (_access(trtFile.c_str(), 0) == 0) {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int fsize = 0;
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) {
            std::cout << "读入模型 " << trtFile << " 失败\n";
            return;
        } else{
            std::cout << "读入模型 " << trtFile << " 成功\n";
        }
        
        /* 创建 推理引擎 */
        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) {
            std::cout << "加载模型失败\n";
            return;
        } else{
            std::cout << "加载模型成功\n";
        }
    }
    /* 创建 序列化网络模型, 并保存至文件 */ 
    else {
        /* 创建 builder */ 
        IBuilder* builder = createInferBuilder(gLogger);
        /* 创建 network */ 
        INetworkDefinition* network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        /* 创建 Optimization Profile, 用于优化 Dynamic Shape */ 
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        /* 创建 BuidlerConfig, 用于设置网络的元数据 */ 
        IBuilderConfig       *config  = builder->createBuilderConfig();
        // 设置 memory pool, 标记为 workspace, 默认为GPU内存
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        /* 设置 网络的输入张量 */ 
        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        // 设置 输入张量的动态范围
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        // 设置 Optimization Profile
        config->addOptimizationProfile(profile);

        /* 设置 网络模型: 此处只有一个 identity layer (output = input)*/
        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        // 标记网络的输出
        network->markOutput(*identityLayer->getOutput(0));
        
        /* 创建 序列化网络模型*/
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0) {
            std::cout << "创建 序列化网络模型 失败\n";
            return;
        } else{
            std::cout << "创建 序列化网络模型 成功\n";
        }

        /* 保存 序列化网络模型 */
        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile) {
            std::cout << "文件 " << trtFile << " 打开失败\n";
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail()) {
            std::cout << "保存 序列化网络模型 失败\n";
            return;
        } else{
            std::cout << "保存 序列化网络模型 成功, 模型保存至 " << trtFile << "\n";
        }

        /* 创建 推理引擎 */
        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) {
            std::cout << "创建 推理引擎 失败\n";
            return;
        } else {
            std::cout << "创建 推理引擎 成功\n";
        }
    }


    /* 查询 引擎的输入输出张量的名称&个数, 并保存下来 */
    long unsigned int nIO     = engine->getNbIOTensors();
    long unsigned int nInput  = 0;
    long unsigned int nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    /* 创建 引擎执行上下文 (GPU context or CPU process) */
    IExecutionContext *context = engine->createExecutionContext();
    // 设置 输入张量的实际大小
    context->setInputShape(vTensorName[0].c_str(), Dims32 {3, {3, 4, 5}});
    // print 输入输出张量的类型、大小、名称
    for (int i = 0; i < nIO; ++i) {
        std::cout << "第[" << i << "]个Tensor 是 " << std::string(i < nInput ? "Input" : "Output") << "\n";
        std::cout << "  类型: " << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << "\n";
        std::cout << "  模型期望大小: " << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << "\n";
        std::cout << "  实际输入大小: " << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << "\n";
        std::cout << "  名称: " << vTensorName[i] << "\n";
    }

    /* 计算每个buffer的字节数 */
    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i) {
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j) {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    /* 创建 CPU和GPU的buffer */
    std::vector<void *> vBufferHost {nIO, nullptr};
    std::vector<void *> vBufferDevice {nIO, nullptr};
    for (int i = 0; i < nIO; ++i) {
        vBufferHost[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferDevice[i], vTensorSize[i]));
    }

    /* 创建输入数据 */
    float *pData = (float *)vBufferHost[0];
    for (int i = 0; i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str())); ++i) {
        pData[i] = float(i);
    }

    /* 将输入数据从CPU拷贝到GPU */
    for (int i = 0; i < nInput; ++i) {
        CHECK(cudaMemcpy(vBufferDevice[i], vBufferHost[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    /* 设置GPU端输入和输出的地址 */
    for (int i = 0; i < nIO; ++i) {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferDevice[i]);
    }

    /* 执行推理 */
    context->enqueueV3(0);

    /* 将输出数据从GPU拷贝到CPU */
    for (int i = nInput; i < nIO; ++i) {
        CHECK(cudaMemcpy(vBufferHost[i], vBufferDevice[i], vTensorSize[i], cudaMemcpyDeviceToHost));
    }

    /* print 输出数据 */
    for (int i = 0; i < nIO; ++i) {
        printArrayInformation((float *)vBufferHost[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, true);
    }

    /* 释放内存 */
    for (int i = 0; i < nIO; ++i) {
        delete[] (char *)vBufferHost[i];
        CHECK(cudaFree(vBufferDevice[i]));
    }
}

int main() {
    CHECK(cudaSetDevice(0));
    // run();  // 创建 序列化网络模型
    run();  // 加载 序列化网络模型, 并进行推理
    return 0;
}
