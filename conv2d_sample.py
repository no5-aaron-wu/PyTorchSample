# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : conv2d_sample.py
@Project  : PyTorch
@Time     : 2021/10/25 17:48
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/10/25 17:48        1.0             None
"""
import torch

class Conv2dModel(torch.nn.Module):
    def __init__(self):
        super(Conv2dModel, self).__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, kernel_size=(3, 3))

    def forward(self, x):
        y = self.conv2d(x)
        return y

def inference(dummy_input, model):
    output = model(dummy_input)
    print(output)


if __name__ == '__main__':
    net = Conv2dModel()
    N = 1
    C = 1
    H = 8
    W = 8
    dummy_input = torch.randn(N, C, H, W)

    #
    inference(dummy_input, net)
    model_name = './conv2dModel.onnx'
    torch.onnx.export(net, dummy_input, model_name, input_names=["input"], output_names=["output"])

