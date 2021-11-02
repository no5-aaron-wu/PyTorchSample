# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : roi_align_sample.py
@Project  : PyTorch
@Time     : 2021/11/1 10:43
@Author   : aaron-wu
@Contact_1: no5aaron@163.com
@Contact_2: wuhao@insta360.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/1 10:43        1.0             None
"""
import numpy
import torch
from torchvision.ops import roi_align
import math
from torch.autograd import Function

torch.set_printoptions(precision=4, sci_mode=False)


def reference_roi_pool(feats, rois, output_size=[7, 7], spatial_scale=1 / 16, sampling_ratio=-1, pool_mode=0,
                       aligned=False):
    x1 = rois[0][1] * spatial_scale
    y1 = rois[0][2] * spatial_scale
    x2 = rois[0][3] * spatial_scale
    y2 = rois[0][4] * spatial_scale

    roi_width = max((x2 - x1), 1)
    roi_height = max((y2 - y1), 1)

    bin_size_w = roi_width / output_size[1]
    bin_size_h = roi_height / output_size[0]

    output = torch.zeros(output_size)

    sampling_ratio_w = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_width / output_size[1])
    sampling_ratio_h = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_height / output_size[0])

    for h in range(output_size[0]):
        for w in range(output_size[1]):
            res = torch.zeros((sampling_ratio_w, sampling_ratio_h))
            for i in range(sampling_ratio_w):
                for j in range(sampling_ratio_h):
                    point_x = x1 + w * bin_size_w + (bin_size_w * (1 + 2 * i) / (2 * sampling_ratio_w))
                    point_y = y1 + h * bin_size_h + (bin_size_h * (1 + 2 * j) / (2 * sampling_ratio_h))
                    if aligned:
                        point_x = max(point_x - 0.5, 0)
                        point_y = max(point_y - 0.5, 0)

                    val0 = feats[0][0][int(point_y)][int(point_x)]
                    val1 = feats[0][0][int(point_y)][int(point_x) + 1]
                    val2 = feats[0][0][int(point_y) + 1][int(point_x)]
                    val3 = feats[0][0][int(point_y) + 1][int(point_x) + 1]

                    dx0 = point_x - int(point_x)
                    dx1 = int(point_x) + 1 - point_x
                    dy0 = point_y - int(point_y)
                    dy1 = int(point_y) + 1 - point_y

                    area0 = dx0 * dy0
                    area1 = dx1 * dy0
                    area2 = dx0 * dy1
                    area3 = dx1 * dy1
                    res[i][j] = val0 * area3 + val1 * area2 + val2 * area1 + val3 * area0
            if pool_mode == 0:  # ave pool
                output[h][w] = torch.mean(res)
            elif pool_mode == 1:  # max pool
                output[h][w] = torch.max(res)

    return output


class RoiAlignFunc(Function):
    @staticmethod
    def forward(ctx, input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
        res = roi_align(input=input, boxes=boxes, output_size=output_size, spatial_scale=spatial_scale,
                        sampling_ratio=sampling_ratio, aligned=aligned)
        return res

    @staticmethod
    def symbolic(g, input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
        input_tensors = [input, boxes]
        return g.op("torch::ROIAlign", *input_tensors, output_size_i=output_size, spatial_scale_f=spatial_scale,
                    sampling_ratio_i=sampling_ratio, aligned_i=aligned)


RoiAlign = RoiAlignFunc.apply


class RoiAlignModel(torch.nn.Module):
    def __init__(self, kernel_size, spatial_scale, sampling_ratio=-1, aligned=False):
        super(RoiAlignModel, self).__init__()
        self.output_size = [kernel_size, kernel_size]
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.roi_align = RoiAlign

    def forward(self, feats, rois):
        r = self.roi_align(feats, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)
        return r


if __name__ == '__main__':
    N = 1
    C = 1
    H = 16
    W = 16
    spatial_scale = 1 / 16
    kernel_size = 7
    pool_mode = 0
    sampling_ratio = 2
    aligned = True#False

    feature_maps = torch.randn(N, C, H, W)
    print(feature_maps)

    rois_in_feature = torch.Tensor([0, 5, 10, 10, 15])
    rois_in_feature = rois_in_feature.unsqueeze(0)

    rois_in_image = rois_in_feature / spatial_scale
    ref_output = reference_roi_pool(feature_maps, rois_in_image, output_size=[kernel_size, kernel_size],
                                    spatial_scale=spatial_scale,
                                    pool_mode=pool_mode, sampling_ratio=sampling_ratio, aligned=aligned)
    print(ref_output)

    # torch_output = roi_align(feature_maps, rois_in_image, output_size=(kernel_size, kernel_size), spatial_scale=spatial_scale,
    #                          sampling_ratio=sampling_ratio, aligned=aligned)
    # print(torch_output)
    # net = RoiAlignModel(kernel_size, spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)
    net = RoiAlignModel(kernel_size, spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)
    torch_output = net(feature_maps, rois_in_image)
    print(torch_output)

    # export
    model_name = './ROIAlignModel.onnx'
    torch.onnx.export(net, (feature_maps, rois_in_image), model_name, input_names=['feats', 'rois'],
                      output_names=['output'])

    pass
