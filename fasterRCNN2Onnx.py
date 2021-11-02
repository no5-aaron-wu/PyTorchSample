# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : fasterRCNN2Onnx.py
@Project  : PyTorch
@Time     : 2021/10/30 17:04
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/10/30 17:04        1.0             None
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch

if __name__ == '__main__':
    num_classes  = 10
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.load_state_dict(torch.load('/content/drive/MyDrive/images/fasterrcnn_resnet50_fpn_9.pth'))
    model.eval()

    # set device to cpu
    cpu_device = torch.device('cpu')
    x = [torch.randn((3, 384, 384), device=cpu_device)]
    model.to(cpu_device)

    # finally convert pytorch model to onnx
    torch.onnx.export(model, x, "faster_rcnn_9.onnx", verbose=True, do_constant_folding=True, opset_version=11)

    pass
