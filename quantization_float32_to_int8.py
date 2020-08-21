import torch
import numpy as np

model_path="/home/xujh/code/pytorch_quantization_demo/quantize_4_cnn_pruning_test.pt"
load_quantization_model=torch.load(model_path)
new_dict={}
for k,v in load_quantization_model.items():
    v1=torch.tensor(v,dtype=torch.int8)
    new_dict[k]=v1
torch.save(new_dict, 'quantize_int4.pt')


