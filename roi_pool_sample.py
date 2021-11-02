# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : roi_pool_sample.py
@Project  : PyTorch
@Time     : 2021/10/25 14:26
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/10/25 14:26        1.0             None
"""
from abc import ABC

import torch
import math
from torch.autograd import Function
from torchvision import ops
import torchvision
from tensorflow.keras.preprocessing.image import load_img
import torchvision.transforms as T


torch.set_printoptions(precision=4, sci_mode=False)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def self_roi_pool(feats, rois):
    # self 计算
    rois = rois.squeeze(0).squeeze(0)
    roi_start_w = int(rois[0][1] + 0.5)
    roi_start_h = int(rois[0][2] + 0.5)
    roi_end_w = int(rois[0][3] + 0.5)
    roi_end_h = int(rois[0][4] + 0.5)

    roi_width = (roi_end_w - roi_start_w + 1)
    roi_height = (roi_end_h - roi_start_h + 1)

    bin_size_w = roi_width / 7
    bin_size_h = roi_height / 7

    self_result = torch.zeros((7, 7))
    for j in range(7):
        for i in range(7):
            self_result[j][i] = (feats[...,
                                 int(roi_start_h + bin_size_h * j):math.ceil(roi_start_h + bin_size_h * (j + 1)),
                                 int(roi_start_w + bin_size_w * i):math.ceil(roi_start_w + bin_size_w * (i + 1))][0][
                                     0].max().item())
    print(self_result)
    pass


class RoiPoolFunc(Function):
    @staticmethod
    def forward(ctx, input, boxes, output_size, spatial_scale):
        res = ops.roi_pool(input=input, boxes=boxes, output_size=output_size, spatial_scale=spatial_scale)
        return res

    @staticmethod
    def symbolic(g, input, boxes, output_size, spatial_scale):
        input_tensors = [input, boxes]
        return g.op("torch::ROIPooling", *input_tensors, output_size_i=output_size, spatial_scale_f=spatial_scale)


roi_pool = RoiPoolFunc.apply


class RoIPoolModel(torch.nn.Module):
    def __init__(self, kernel_size=7, spatial_scale=1 / 16):
        super(RoIPoolModel, self).__init__()
        self.output_size = [kernel_size, kernel_size]
        self.spacial_scale = spatial_scale
        self.roi_pool = roi_pool

    def forward(self, feats, rois):
        rois = rois.squeeze(0).squeeze(0)
        r = self.roi_pool(feats, rois, self.output_size, self.spacial_scale)
        return r

class RoIPoolWithRPNModel(torch.nn.Module):
    def __init__(self, img_height=800, img_width=800, kernel_size=7):
        super(RoIPoolWithRPNModel, self).__init__()
        self.targets=None
        self.transform = model.transform
        self.backbone = model.backbone
        self.model_rpn = model.rpn
        self.rpn = self.model_rpn
        self.roi_pool = roi_pool
        self.img_height = img_height
        self.img_width = img_width
        self.output_size = [kernel_size, kernel_size]
        self.spatial_scale = 1/16
        pass
    def forward(self, images):
        images, self.targets = self.transform(images, self.targets)
        features = self.backbone(images.tensors)
        prorosals, _ = self.rpn(images, features, self.targets)
        third_level_feature = features['2']
        # third_level_feature = features['2'][:, 0, :, :]
        # third_level_feature = third_level_feature.unsqueeze(1)
        self.spatial_scale = float(third_level_feature.size()[2] / self.img_height)
        test_proposal = prorosals[0][0]
        rois_in_image = torch.cat((torch.zeros(1), test_proposal), dim=0)
        rois_in_image = rois_in_image.unsqueeze(0)
        r = self.roi_pool(third_level_feature, rois_in_image, self.output_size, self.spatial_scale)

        return r
        pass


def inference(feats, rois, model):
    output = model(feats, rois)
    print(output)

def case1():
    N = 1
    C = 1
    H = 16
    W = 16
    feature_maps = torch.randn(N, C, H, W)
    print("========================")
    print(feature_maps)
    rois_in_feature = torch.Tensor([0, 5, 10, 10, 15])
    # rois_in_feature = rois_in_feature.unsqueeze(0)
    rois_in_feature = rois_in_feature.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    inv_spatial_scale = 16
    rois_in_image = rois_in_feature * inv_spatial_scale

    self_roi_pool(feature_maps, rois_in_feature)

    net = RoIPoolModel()
    inference(feature_maps, rois_in_image, net)

    # export
    model_name = './SqueezeROIPoolingModel.onnx'
    torch.onnx.export(net, (feature_maps, rois_in_image), model_name,
                      input_names=['feats', 'rois'], output_names=['output'],
                      # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      # opset_version=11
                      )
    # import and run
    # sess = rt.InferenceSession("roiPoolModel.onnx")
    # outputs = ['output']
    # result = sess.run(outputs, {'feats': feature_maps, 'rois': rois_in_image})
    pass
def case2():
    img = load_img('2007_000032.jpg', target_size=(800, 800))
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    images = [img]

    net = RoIPoolWithRPNModel()
    output = net(images)
    print(output)
    # export
    model_name = './RPNROIPoolingModel.onnx'
    torch.onnx.export(net, images, model_name,
                      input_names=['images'], output_names=['output'], opset_version=11)



    pass

if __name__ == '__main__':
    # case 1
    # case1()
    # case 2
    case2()


    pass
