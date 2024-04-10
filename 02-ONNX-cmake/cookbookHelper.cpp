#include "cookbookHelper.cuh"

using namespace nvinfer1;

// 文件路径
const std::string model_file {"./model.plan"};
// 创建 Logger, 包含的日志等级为: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
static Logger gLogger(ILogger::Severity::kERROR);

class OnnxModel{
public: 
    OnnxModel(const std::string file) {
        valid = load_model_from_file(file);
        build_model_metadata();
        create_context();
    }

    ~OnnxModel(){
        /* 释放内存 */
        for (int i = 0; i < nIO; ++i) {
            delete[] (char *)vBufferHost[i];
            CHECK(cudaFree(vBufferDevice[i]));
        }
    }

    void infer() {
        std::cout << "*********************************************\n";
        std::cout << "模型推理:\n";
        
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
        std::cout << "*********************************************\n";
    }

    void* GetBuffer(int index, int& size, nvinfer1::Dims32& dim, std::string& name){
        if (index >= nIO) return nullptr;
        size = vTensorSize[index];
        dim = context->getTensorShape(vTensorName[index].c_str());
        name = vTensorName[index];
        return vBufferHost[index];
    }
    
private:
    /* 模型对应的推理引擎 */
    ICudaEngine* engine = nullptr;

    /* 模型是否创建成功 */
    bool valid = false; 
    
    /* 模型的 IO tensor 信息 */
    long unsigned int nIO     = 0;          // IO tensor 数
    long unsigned int nInput  = 0;          // input tensor 数
    long unsigned int nOutput = 0;          // output tensor 数
    std::vector<std::string> vTensorName;   // IO tensor 名称

    /* 模型的上下文 */
    IExecutionContext* context = nullptr;
    std::vector<int> vTensorSize;       // 每个 Tensor 的大小
    std::vector<void *> vBufferHost;    // CPU Buffer
    std::vector<void *> vBufferDevice;  // GPU Buffer

private:
    bool load_model_from_file(const std::string file) {
        std::cout << "*********************************************\n";
        std::cout << "载入 .plan 序列化网络模型:\n";
        if (_access(file.c_str(), 0) != 0) return false;
         
        std::ifstream engineFile(file, std::ios::binary);
        long int fsize = 0;
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) {
            std::cout << "读入模型文件:" << file << " 失败\n";
            return false;
        } else{
            std::cout << "读入模型文件:" << file << " 成功\n";
        }
        
        /* 创建 推理引擎 */
        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) {
            std::cout << "加载模型失败\n";
            return false;
        } else{
            std::cout << "加载模型成功\n";
        }
        std::cout << "*********************************************\n";
        return true;
    }

    void build_model_metadata(){
        std::cout << "*********************************************\n";
        std::cout << "构建网络的元数据:\n";
        if(!valid) { std::cout << "模型未成功创建!\n"; return; }

        nIO = engine->getNbIOTensors();
        vTensorName.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            vTensorName[i] = std::string(engine->getIOTensorName(i));
            nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
            nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
        }
        std::cout << "*********************************************\n";
    }

    void create_context() {
        std::cout << "*********************************************\n";
        std::cout << "创建引擎执行上下文:\n";
        if(!valid) { std::cout << "模型未成功创建!\n"; return; }
        
        context = engine->createExecutionContext();
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
        vTensorSize.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
            int    size = 1;
            for (int j = 0; j < dim.nbDims; ++j) 
                size *= dim.d[j];
            vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
        }

        /* 创建 CPU和GPU的 buffer */
        vBufferHost.resize(nIO);
        vBufferDevice.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            vBufferHost[i] = (void *)new char[vTensorSize[i]];
            CHECK(cudaMalloc(&vBufferDevice[i], vTensorSize[i]));
        }
        std::cout << "*********************************************\n";
    }
};

int main() {
    CHECK(cudaSetDevice(0));
    OnnxModel model(model_file);
    
    int buffer_size = 0;
    nvinfer1::Dims32 buffer_dim;
    std::string buffer_name;
    float *buffer_data = nullptr;

    /* 设置输入数据 */
    buffer_data = (float*)model.GetBuffer(0, buffer_size, buffer_dim, buffer_name);
    for (int i = 0; i < buffer_size / dataTypeToSize(nvinfer1::DataType::kFLOAT); ++i) {
        buffer_data[i] = float(i);
    }

    model.infer();

    /* print tensor */
    for (int i = 0; i < 1; ++i) {
        buffer_data = (float*)model.GetBuffer(i, buffer_size, buffer_dim, buffer_name);
        printArrayInformation(buffer_data, buffer_dim, buffer_name , true, true);
    }
    return 0;
}
