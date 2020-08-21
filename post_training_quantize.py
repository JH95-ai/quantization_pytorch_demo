from model import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
from collections import OrderedDict
import pruning_quantization

def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 200 == 0:
            break
    print('direct quantization finish')


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

#量化函数
def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        '''   if True:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        torch.save(model.state_dict(), 'ckpt/quantize_8_cnn.pt')'''
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    batch_size = 64
    save_model=True
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    model = Net()
    model.load_state_dict(torch.load('ckpt/mnist_cnn.pt'))
    model.eval()
    full_inference(model, test_loader)
    num_bits = 8
    model.quantize(num_bits=num_bits)
    print('Quantization bit: %d' % num_bits)
    #求网络中各层的qi,qo,qw.
    direct_quantize(model, train_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #固定求得的变量值
    model.freeze()
    quantize_inference(model, test_loader)
    if True:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        torch.save(model.state_dict(), 'ckpt/quantize_8_cnn_5.pt')
