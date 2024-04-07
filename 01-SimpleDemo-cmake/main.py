import os
import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = "./model.plan"
# 输入数据
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

def run():
    # 创建 Logger, 包含的日志等级为: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
    logger = trt.Logger(trt.Logger.ERROR)
    
    # 载入 .plan 序列化网络模型
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print(f"加载模型 {trtFile} 失败")
            return
        else:
            print(f"加载模型 {trtFile} 成功")
    # 创建 序列化网络模型, 并保存至文件
    else:
        # 创建 builder
        builder = trt.Builder(logger)
        # 创建 network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 创建 Optimization Profile, 用于优化 Dynamic Shape
        profile = builder.create_optimization_profile()
        # 创建 BuidlerConfig, 用于设置网络的元数据
        config = builder.create_builder_config()
        # 设置 memory pool, 标记为 workspace, 默认为GPU内存
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # 设置 网络的输入张量
        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        # 设置 输入张量的动态范围
        profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
        # 设置 Optimization Profile
        config.add_optimization_profile(profile)

        # 设置 网络模型: 此处只有一个 identity layer (output = input)
        identityLayer = network.add_identity(inputTensor)
        # 标记网络的输出
        network.mark_output(identityLayer.get_output(0))

        # 创建 序列化网络模型
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("创建 序列化网络模型 失败")
            return
        else:
            print("创建 序列化网络模型 成功")
        # 保存 序列化网络模型
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print(f"保存 序列化网络模型 成功, 模型保存至 {trtFile}")

    # 创建 推理引擎
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("创建 推理引擎 失败")
        return
    else:
        print("创建 推理引擎 成功")

    # 查询 引擎的输入输出张量的名称&个数, 并保存下来
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    #nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

    # 创建 引擎执行上下文 (GPU context or CPU process)
    context = engine.create_execution_context()
    # 设置 输入张量的实际大小
    context.set_input_shape(lTensorName[0], [3, 4, 5])
    # print 输入输出张量的类型、大小、名称
    for i in range(nIO):
        print("第[{0}]个Tensor, 是 {1}".format(i, "Input" if i < nInput else "Output"))
        print(f"  类型: {engine.get_tensor_dtype(lTensorName[i])}")
        print(f"  模型期望大小: {engine.get_tensor_shape(lTensorName[i])}")
        print(f"  实际输入大小: {context.get_tensor_shape(lTensorName[i])}")
        print(f"  名称: {lTensorName[i]}")

    # 创建 CPU和GPU的buffer
    bufferHost = []
    bufferHost.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferHost.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferDevice = []
    for i in range(nIO):
        bufferDevice.append(cudart.cudaMalloc(bufferHost[i].nbytes)[1])

    # 将输入数据从CPU拷贝到GPU
    for i in range(nInput): 
        cudart.cudaMemcpy(bufferDevice[i], bufferHost[i].ctypes.data, bufferHost[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # 设置GPU端输入和输出的地址
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferDevice[i]))

    # 执行推理
    context.execute_async_v3(0)

    # 将输出数据从GPU拷贝到CPU
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferHost[i].ctypes.data, bufferDevice[i], bufferHost[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # print 输出数据
    for i in range(nIO):
        print("tensor 名称: ", lTensorName[i])
        print(bufferHost[i])
        print("")

    # 释放内存
    for b in bufferDevice:
        cudart.cudaFree(b)

if __name__ == "__main__":
    if os.path.exists(trtFile):
        run() # 加载 序列化网络模型, 并进行推理
    else:
        run() # 创建 序列化网络模型
        run() # 加载 序列化网络模型, 并进行推理
