import torch

model_path="ckpt/quantize_8_cnn.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_state = torch.load(model_path,map_location=device)

pretrained_dict = {k: v for k, v in weight_state.items() }
layer_name=list(pretrained_dict.keys())
tensor=list(pretrained_dict.values())
new_dict={}
for i in range(len(layer_name[6:])):
    new_dict[layer_name[i]]=tensor[i]

torch.save(new_dict, 'quantize_8_cnn_pruning_test_2.pt')
'''del pretrained_dict['conv1.weight']
del pretrained_dict['conv2.weight']
del pretrained_dict['fc.weight']
del pretrained_dict['conv1.bias']
del pretrained_dict['conv2.bias']
del pretrained_dict['fc.bias']

torch.save(pretrained_dict, 'quantize_8_cnn_pruning_test.pt')'''




