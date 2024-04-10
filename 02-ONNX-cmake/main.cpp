#include "TensorRTUtils.h"

/* TensorRT context */
const std::string model_path = "./model/weight-epoch-99-loss-0.010884195198304952.onnx";
static TensorRTUtils tensorRT_utils(model_path, 16, 12);
static float width = 512, height = 512;

void compile(){

}

void execute(){
    tensorRT_utils.getTensorRTContext()->enqueueV3
}

int main(){
    compile();
    execute();
    return 0;
}