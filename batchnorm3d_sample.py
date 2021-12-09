# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : batchnorm3d_sample.py
@Project  : PyTorchSample
@Time     : 2021/12/2 18:29
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/12/2 18:29        1.0             None
"""
import torch
import numpy as np

torch.set_printoptions(precision=4, sci_mode=False)


def reference_bn2d(data, mean, var, scale, bias, eps):
    shape = data.shape
    C = shape[1]

    output = torch.zeros(shape)
    for c in range(C):
        x = data[:, c, :, :]
        # x_np = np.array(x)
        # x_mean = np.mean(x_np)
        # x_var = np.var(x_np)
        # print(x_mean, x_var, x.var() / (H * W) * (H * W - 1))
        output[:, c, :, :] = ((x - mean[c]) / (var[c] + eps) ** (1 / 2)) * scale[c] + bias[c]
    return output
    pass


def reference_bn3d(data, mean, var, scale, bias, eps):
    shape = data.shape
    C = shape[1]

    output = torch.zeros(shape)
    for c in range(C):
        x = data[:, c, :, :, :]
        output[:, c, :, :, :] = ((x - mean[c]) / (var[c] + eps) ** (1 / 2)) * scale[c] + bias[c]
    return output


class BatchNorm3DModel(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm3DModel, self).__init__()
        self.bn3d = torch.nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine,
                                         track_running_stats=track_running_stats)
        self.bn3d.weight = torch.nn.Parameter(0.9 * torch.ones(num_features))
        self.bn3d.bias = torch.nn.Parameter(0.1 * torch.ones(num_features))

    def forward(self, x):
        y = self.bn3d(x)
        return y


class BatchNorm2DModel(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2DModel, self).__init__()
        self.bn2d = torch.nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                         track_running_stats=track_running_stats)

        self.bn2d.weight = torch.nn.Parameter(0.9 * torch.ones(num_features))
        self.bn2d.bias = torch.nn.Parameter(0.1 * torch.ones(num_features))

    def forward(self, x):
        y = self.bn2d(x)
        return y


if __name__ == '__main__':
    N = 2
    C = 2
    D = 2
    H = 4
    W = 4

    # input_data = torch.randn(N, C, D, H, W)
    input_data = torch.Tensor([[[[[-0.0719, -1.1485, -0.8121, 0.3210],
                                  [0.7045, -1.0218, -0.1079, -0.0445],
                                  [-1.0393, -0.1736, -0.2990, -1.0323],
                                  [0.4475, -1.0452, -0.7920, 0.8701]],

                                 [[1.0191, -1.0985, -1.2718, 1.7572],
                                  [-0.2655, 1.1289, 0.1451, -2.5228],
                                  [-0.5156, 0.3338, 0.3000, -0.7299],
                                  [0.8030, 0.2498, -2.1443, -1.1359]]],

                                [[[-1.0552, 0.6933, 1.6670, 1.6444],
                                  [-1.2459, -0.0375, 1.8998, 0.7607],
                                  [-0.5371, 0.5064, 0.1969, -0.8389],
                                  [0.6349, 0.0928, -1.1692, 0.2719]],

                                 [[0.6792, -0.4264, -0.8778, -1.0527],
                                  [-1.5256, -0.3726, -1.4898, -1.1416],
                                  [0.2455, 0.8780, -0.6516, -1.3824],
                                  [0.3578, 0.3000, 0.4153, 0.6024]]]],

                               [[[[-2.2028, 2.2720, -0.6464, 0.1423],
                                  [-1.5544, -1.4971, -0.4033, -1.7692],
                                  [-0.6903, 0.7621, 0.1392, 0.3346],
                                  [0.7621, -0.4212, 0.8158, 1.4076]],

                                 [[0.4131, -0.8365, -1.9237, 0.7512],
                                  [0.1862, -1.1820, 1.8072, 0.3006],
                                  [0.9244, -0.2365, -0.7630, -0.7030],
                                  [0.3194, -0.2355, 0.7031, -0.6989]]],

                                [[[0.3913, 0.7548, 0.6100, -0.5915],
                                  [-1.3669, 1.2156, -0.1940, 0.2449],
                                  [2.8526, -0.0430, 0.4407, -0.0945],
                                  [-1.1139, 0.5690, 0.9346, 1.2166]],

                                 [[-1.4685, 0.2177, -0.4763, 0.1243],
                                  [-0.0478, 1.8844, 0.1323, -0.4415],
                                  [0.2545, 0.8754, 1.2402, -0.2936],
                                  [-0.6361, 1.0708, 0.7563, -0.4433]]]]])
    # input_data_2d = torch.randn(N, C * D, H, W)
    # input_data_2d = torch.arange(1, N * C * D * H * W + 1, dtype=torch.float32).reshape([N, C * D, H, W])
    input_data_2d = input_data.reshape([N, C * D, H, W])

    print("input_data_2d: ", input_data_2d.shape)
    print(input_data_2d)
    print("input_data: ", input_data.shape)
    print(input_data)

    bn2d_net = BatchNorm2DModel(num_features=C * D, affine=True, momentum=0.1)
    bn3d_net = BatchNorm3DModel(num_features=C)

    bn2d_output = bn2d_net(input_data_2d)
    bn3d_output = bn3d_net(input_data)
    bn2d_net.eval()
    bn3d_net.eval()
    bn2d_output = bn2d_net(input_data_2d)
    bn3d_output = bn3d_net(input_data)

    ref_bn2d_output = reference_bn2d(input_data_2d, bn2d_net.bn2d.running_mean.data, bn2d_net.bn2d.running_var.data,
                                     bn2d_net.bn2d.weight, bn2d_net.bn2d.bias, bn2d_net.bn2d.eps)
    ref_bn3d_output = reference_bn3d(input_data, bn3d_net.bn3d.running_mean.data, bn3d_net.bn3d.running_var.data,
                                     bn3d_net.bn3d.weight, bn3d_net.bn3d.bias, bn3d_net.bn3d.eps)

    print("bn2d output: ", bn2d_output.shape)
    print(bn2d_output)
    print("bn3d output: ", bn3d_output.shape)
    print(bn3d_output)
    print("ref2d output: ", ref_bn2d_output.shape)
    print(ref_bn2d_output)
    print("ref3d output: ", ref_bn3d_output.shape)
    print(ref_bn3d_output)

    print(torch.isclose(ref_bn2d_output, bn2d_output, atol=1e-03).all())
    print(torch.isclose(ref_bn3d_output, bn3d_output, atol=1e-03).all())

    model_name = 'D:/code/MNN/build/tools/converter/BatchNorm2DModel.onnx'
    torch.onnx.export(bn2d_net, input_data_2d, model_name, input_names=["input"], output_names=["output"],
                      opset_version=11)
    model_name = 'D:/code/MNN/build/tools/converter/BatchNorm3DModel.onnx'
    torch.onnx.export(bn3d_net, input_data, model_name, input_names=["input"], output_names=["output"],
                      opset_version=11)

    pass
