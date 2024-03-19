import sys
import torch
import tensorrt as trt
print('python: ', sys.version)
print('torch: ', torch.__version__)
print('cuda: ', torch.version.cuda)
print('cudnn: ', torch.backends.cudnn.version())
print('tensorRT: ',  trt.__version__)