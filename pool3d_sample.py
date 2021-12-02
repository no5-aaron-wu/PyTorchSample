# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : pool3d_sample.py
@Project  : PyTorch
@Time     : 2021/11/22 16:01
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/22 16:01        1.0             None
"""
import torch
import onnx
from onnxsim import simplify

class MaxPool2DModel(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=False):
        super(MaxPool2DModel, self).__init__()
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        y = self.maxpool2d(x)
        return y


class AvgPool2DModel(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=False):
        super(AvgPool2DModel, self).__init__()
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        y = self.avgpool2d(x)
        return y


class MaxPool3DModel(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), ceil_mode=False):
        super(MaxPool3DModel, self).__init__()
        self.maxpool3d = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding,
                                            ceil_mode=ceil_mode)

    def forward(self, x):
        y = self.maxpool3d(x)
        return y


class AvgPool3DModel(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), ceil_mode=False):
        super(AvgPool3DModel, self).__init__()
        self.avgpool3d = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding,
                                            ceil_mode=ceil_mode)

    def forward(self, x):
        y = self.avgpool3d(x)
        return y


if __name__ == '__main__':
    N = 1
    C = 2
    D = 3
    H = 4
    W = 4
    # input_data = torch.randn(N, C, D, H, W)
    input_data = torch.Tensor([[[[[-0.7932, 2.4321, 0.0637, -0.4720],
                                  [0.0226, 0.0862, -0.6992, 0.9766],
                                  [0.5793, -0.8382, -0.2175, 0.8473],
                                  [-0.1035, 2.2405, 1.6605, -0.5414]],

                                 [[1.6806, 0.2003, 0.4645, -0.4300],
                                  [-1.5504, 0.3764, 1.4385, 1.3375],
                                  [-0.4558, 0.6160, -0.8035, 1.2687],
                                  [1.1023, -0.1308, -0.8731, -0.9811]],

                                 [[-0.2629, 0.4746, -0.5862, 0.5464],
                                  [0.7505, 0.5210, -0.2518, 1.4507],
                                  [0.5926, -0.1340, -1.3873, 0.0803],
                                  [0.5536, 1.4215, 0.7764, -0.3433]]],

                                [[[-0.7481, -1.7959, 0.4001, 1.0373],
                                  [-0.0043, 0.4523, 1.5205, 1.2470],
                                  [-0.0192, 0.4941, 0.4697, -0.4880],
                                  [0.3149, 0.2813, -0.2804, -0.0743]],

                                 [[0.7062, -0.1173, 1.5354, -0.1380],
                                  [0.5673, 2.9599, -0.2510, 0.8631],
                                  [-0.6965, 0.9290, -0.2999, 0.9968],
                                  [1.2393, -1.6203, -0.1799, -0.1299]],

                                 [[0.4772, 1.3508, -0.1827, 0.3575],
                                  [0.5808, 0.3754, -0.4300, -0.1269],
                                  [-0.4508, -0.7323, 0.5261, 0.2068],
                                  [-0.7043, -0.4302, -0.1138, -0.9379]]]]])
    # input_data_2d = torch.randn(N, C * D, H, W)
    input_data_2d = torch.Tensor([[[[-0.7932, 2.4321, 0.0637, -0.4720],
                                     [0.0226, 0.0862, -0.6992, 0.9766],
                                     [0.5793, -0.8382, -0.2175, 0.8473],
                                     [-0.1035, 2.2405, 1.6605, -0.5414]],

                                    [[1.6806, 0.2003, 0.4645, -0.4300],
                                     [-1.5504, 0.3764, 1.4385, 1.3375],
                                     [-0.4558, 0.6160, -0.8035, 1.2687],
                                     [1.1023, -0.1308, -0.8731, -0.9811]],

                                    [[-0.2629, 0.4746, -0.5862, 0.5464],
                                     [0.7505, 0.5210, -0.2518, 1.4507],
                                     [0.5926, -0.1340, -1.3873, 0.0803],
                                     [0.5536, 1.4215, 0.7764, -0.3433]],

                                   [[-0.7481, -1.7959, 0.4001, 1.0373],
                                     [-0.0043, 0.4523, 1.5205, 1.2470],
                                     [-0.0192, 0.4941, 0.4697, -0.4880],
                                     [0.3149, 0.2813, -0.2804, -0.0743]],

                                    [[0.7062, -0.1173, 1.5354, -0.1380],
                                     [0.5673, 2.9599, -0.2510, 0.8631],
                                     [-0.6965, 0.9290, -0.2999, 0.9968],
                                     [1.2393, -1.6203, -0.1799, -0.1299]],

                                    [[0.4772, 1.3508, -0.1827, 0.3575],
                                     [0.5808, 0.3754, -0.4300, -0.1269],
                                     [-0.4508, -0.7323, 0.5261, 0.2068],
                                     [-0.7043, -0.4302, -0.1138, -0.9379]]]])

    print("input_data: ", input_data.shape)
    print(input_data)
    print("input_data_2d: ", input_data_2d.shape)
    print(input_data_2d)

    maxpool3d_net = MaxPool3DModel(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), ceil_mode=False)
    avgpool3d_net = AvgPool3DModel(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), ceil_mode=True)
    maxpool2d_net = MaxPool2DModel(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)
    avgpool2d_net = AvgPool2DModel(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=True)

    torch_output = maxpool3d_net(input_data)
    print("maxpool3d output: ", torch_output.shape)
    print(torch_output)

    torch_output = avgpool3d_net(input_data)
    print("avgpool3d output: ", torch_output.shape)
    print(torch_output)

    torch_output = maxpool2d_net(input_data_2d)
    print("maxpool2d output: ", torch_output.shape)
    print(torch_output)

    torch_output = avgpool2d_net(input_data_2d)
    print("avgpool2d output: ", torch_output.shape)
    print(torch_output)

    model_name = './MaxPool3DModel.onnx'
    torch.onnx.export(maxpool3d_net, input_data, model_name, input_names=["input"], output_names=["output"],
                      opset_version=11)

    model_name = './AvgPool3DModel.onnx'
    torch.onnx.export(avgpool3d_net, input_data, model_name, input_names=["input"], output_names=["output"],
                      do_constant_folding=True, opset_version=11)

    model_name = './MaxPool2DModel.onnx'
    torch.onnx.export(maxpool2d_net, input_data_2d, model_name, input_names=["input"], output_names=["output"],
                      opset_version=11)

    model_name = './AvgPool2DModel.onnx'
    torch.onnx.export(avgpool2d_net, input_data_2d, model_name, input_names=["input"], output_names=["output"],
                      opset_version=11)

    avgPool3DModel = onnx.load('./AvgPool3DModel.onnx')
    avgPool3DModelSim, check = simplify(avgPool3DModel)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(avgPool3DModelSim, './AvgPool3DModelSim.onnx')
    pass
