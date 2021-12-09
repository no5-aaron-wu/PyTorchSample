# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : rnn_sample.py
@Project  : PyTorchSample
@Time     : 2021/12/8 18:09
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/12/8 18:09        1.0             None
"""
import torch
import onnx
import onnxruntime


class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input, h_0):
        output, h_n = self.rnn(input, h_0)
        return output, h_n


if __name__ == '__main__':
    input_size = 100
    hidden_size = 20
    num_layers = 4
    seq_len = 10
    batch_size = 1

    input_data = torch.randn(seq_len, batch_size, input_size)
    h0_data = torch.zeros(num_layers, batch_size, hidden_size)

    print("input_data: ", input_data.shape)
    print(input_data)

    rnn_net = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    rnn_output, rnn_h_n = rnn_net(input_data, h0_data)
    # print("rnn output: ", rnn_output.shape)
    print(rnn_output)

    model_name = 'D:/code/MNN/build/tools/converter/RNNModel.onnx'
    torch.onnx.export(rnn_net, (input_data, h0_data), model_name, input_names=["input", "h_0"],
                      output_names=["output", "h_n"],
                      opset_version=11, do_constant_folding=True)

    # onnx_model = onnx.load(model_name)
    # onnx.checker.check_model(model_name)
    # session = onnxruntime.InferenceSession(model_name)

    pass
