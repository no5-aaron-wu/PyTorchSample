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


class PreCalc:
    def __init__(self, pos=[0, 0, 0, 0], w=[0, 0, 0, 0]):
        self.pos = pos
        self.w = w


def pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_start_h, roi_start_w, bin_size_h,
                                      bin_size_w, sampling_ratio_h, sampling_ratio_w):
    pre_calc = []

    sampling_bin_h = bin_size_h / sampling_ratio_h
    sampling_bin_w = bin_size_w / sampling_ratio_w

    for h in range(pooled_height):
        sampling_start_h = roi_start_h + h * bin_size_h
        for w in range(pooled_width):
            sampling_start_w = roi_start_w + w * bin_size_w
            for i in range(sampling_ratio_h):
                py = sampling_start_h + (0.5 + i) * sampling_bin_h
                for j in range(sampling_ratio_w):
                    px = sampling_start_w + (0.5 + j) * sampling_bin_w
                    if py < -1.0 or py > height or px < -1.0 or px > width:
                        pc = PreCalc()
                        pre_calc.append(pc)
                        continue

                    if py < 0:
                        py = 0
                    if px < 0:
                        px = 0

                    py0 = int(py)
                    px0 = int(px)
                    py1 = py0 + 1
                    px1 = px0 + 1

                    if py0 >= height - 1:
                        py0 = py1 = height - 1
                        py = float(py0)
                    if px0 >= width - 1:
                        px0 = px1 = width - 1
                        px = float(px0)

                    dy0 = py - py0
                    dx0 = px - px0
                    dy1 = 1. - dy0
                    dx1 = 1. - dx0
                    area0 = dx0 * dy0
                    area1 = dx1 * dy0
                    area2 = dx0 * dy1
                    area3 = dx1 * dy1
                    pos0 = py0 * width + px0
                    pos1 = py0 * width + px1
                    pos2 = py1 * width + px0
                    pos3 = py1 * width + px1
                    pc = PreCalc(pos=[pos0, pos1, pos2, pos3], w=[area3, area2, area1, area0])
                    pre_calc.append(pc)
    return pre_calc


def reference_roi_align(feats, rois, output_size=[7, 7], spatial_scale=1 / 16, sampling_ratio=-1, pool_mode=0,
                        aligned=False):
    num_roi = rois.size()[0]
    channel = feats.size()[1]
    input_h = feats.size()[2]
    input_w = feats.size()[3]
    output = torch.zeros((num_roi, channel, output_size[0], output_size[1]))

    offset = -0.5 if aligned else 0.0
    for n in range(num_roi):
        batch_index = int(rois[n][0])
        x1 = rois[n][1].item() * spatial_scale + offset
        y1 = rois[n][2].item() * spatial_scale + offset
        x2 = rois[n][3].item() * spatial_scale + offset
        y2 = rois[n][4].item() * spatial_scale + offset

        roi_width = x2 - x1
        roi_height = y2 - y1
        if not aligned:
            roi_width = max(roi_width, 1.)
            roi_height = max(roi_height, 1.)

        bin_size_w = roi_width / output_size[1]
        bin_size_h = roi_height / output_size[0]

        sampling_ratio_w = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_width / output_size[1])
        sampling_ratio_h = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_height / output_size[0])

        pre_calc = pre_calc_for_bilinear_interpolate(input_h, input_w, output_size[0], output_size[1], y1, x1,
                                                     bin_size_h, bin_size_w, sampling_ratio_h, sampling_ratio_w)

        for c in range(channel):
            pre_calc_index = 0
            for h in range(output_size[0]):
                for w in range(output_size[1]):
                    res = torch.zeros((sampling_ratio_h, sampling_ratio_w))
                    for i in range(sampling_ratio_h):
                        for j in range(sampling_ratio_w):
                            pc = pre_calc[pre_calc_index]
                            val0 = feats[batch_index][c][pc.pos[0] // input_w][pc.pos[0] % input_w]
                            val1 = feats[batch_index][c][pc.pos[1] // input_w][pc.pos[1] % input_w]
                            val2 = feats[batch_index][c][pc.pos[2] // input_w][pc.pos[2] % input_w]
                            val3 = feats[batch_index][c][pc.pos[3] // input_w][pc.pos[3] % input_w]

                            res[i][j] = val0 * pc.w[0] + val1 * pc.w[1] + val2 * pc.w[2] + val3 * pc.w[3]
                            pre_calc_index += 1
                    if pool_mode == 0:  # ave pool
                        output[n][c][h][w] = torch.mean(res)
                    elif pool_mode == 1:  # max pool
                        output[n][c][h][w] = torch.max(res)

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
    C = 256
    H = 25
    W = 25
    spatial_scale = 1 / 16
    kernel_size = 7
    pool_mode = 0
    sampling_ratio = 2
    # aligned = True
    aligned = False

    feature_maps = torch.randn(N, C, H, W)
    # feature_maps = torch.Tensor()
    print(feature_maps)

    # rois_in_feature = torch.Tensor([[0, 0, 1, 2, 3], [2, 0.5, 1, 1.5, 2]])
    # rois_in_feature = torch.Tensor([[0, 3.1, 7, 1, 3]])
    # rois_in_feature = rois_in_feature.unsqueeze(0)

    roi_nums = 50
    rois_in_feature = torch.normal(mean=(H + W) / 4, std=(H + W) / 8, size=(roi_nums, 4))
    rois_batch_index = torch.linspace(0, N - 1, steps=roi_nums).unsqueeze(1)
    rois_batch_index = rois_batch_index.floor()
    rois_in_feature = torch.cat([rois_batch_index, rois_in_feature], dim=1)

    print(rois_in_feature)

    rois_in_image = torch.clone(rois_in_feature)
    rois_in_image[:, 1:] /= spatial_scale

    ref_output = reference_roi_align(feature_maps, rois_in_image, output_size=[kernel_size, kernel_size],
                                     spatial_scale=spatial_scale,
                                     pool_mode=pool_mode, sampling_ratio=sampling_ratio, aligned=aligned)
    print(ref_output)

    # torch_output = roi_align(feature_maps, rois_in_image, output_size=(kernel_size, kernel_size),
    #                          spatial_scale=spatial_scale,
    #                          sampling_ratio=sampling_ratio, aligned=aligned)
    # print(torch_output)
    net = RoiAlignModel(kernel_size, spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)
    torch_output = net(feature_maps, rois_in_image)
    print(torch_output)

    print(torch.isclose(ref_output, torch_output))
    print(torch.isclose(ref_output, torch_output, atol=1e-03).all())

    # export
    model_name = './ROIAlignModel_1x256x25x25_50_1d16_k7p0s2a0.onnx'
    torch.onnx.export(net, (feature_maps, rois_in_image), model_name, input_names=['feats', 'rois'],
                      output_names=['output'])

    pass
