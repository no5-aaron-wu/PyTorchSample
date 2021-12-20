# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : lstm_sample.py
@Project  : PyTorchSample
@Time     : 2021/12/9 10:26
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/12/9 10:26        1.0             None
"""
import torch


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input, h_0, c_0):
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return output, h_n, c_n


class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input, h_0):
        output, h_n = self.gru(input, h_0)
        return output, h_n


if __name__ == '__main__':
    input_size = 5
    hidden_size = 2
    num_layers = 1
    seq_len = 10
    batch_size = 1

    torch.manual_seed(0)
    input_data = torch.randn(seq_len, batch_size, input_size)
    h_0_data = torch.zeros(num_layers, batch_size, hidden_size)
    c_0_data = torch.zeros(num_layers, batch_size, hidden_size)

    print("input_data shape: ", input_data.shape)
    print(input_data)
    print("h0_data shape: ", h_0_data.shape)
    print("c0_data shape: ", c_0_data.shape)

    lstm_net = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    gru_net = GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    lstm_output, lstm_h_n, lstm_c_n = lstm_net(input_data, h_0_data, c_0_data)
    gru_output, gru_h_n = gru_net(input_data, h_0_data)

    print("lstm_output shape: ", lstm_output.shape)
    print(lstm_output)
    print("lstm_h_n shape: ", lstm_h_n.shape)
    print(lstm_h_n)
    print("lstm_c_n shape: ", lstm_c_n.shape)
    print(lstm_c_n)

    print("gru_output shape: ", gru_output.shape)
    print(gru_output)

    model_name = 'D:/code/MNN/build/tools/converter/LSTMModel.onnx'
    torch.onnx.export(lstm_net, (input_data, h_0_data, c_0_data), model_name, input_names=["input", "h_0", "c_0"],
                      output_names=["output", "h_n", "c_n"],
                      opset_version=11, do_constant_folding=True)
    model_name = 'D:/code/MNN/build/tools/converter/GRUModel.onnx'
    torch.onnx.export(gru_net, (input_data, h_0_data), model_name, input_names=["input", "h_0"],
                      output_names=["output", "h_n"],
                      opset_version=11, do_constant_folding=True)

    pass
