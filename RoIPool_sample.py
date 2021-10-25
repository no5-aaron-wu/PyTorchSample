import torch
import torchvision
import math
# from torchvision.ops import RoIPool

def self_roi_pool(feat, rois_in_feature):
    # self 计算
    roi_start_w = int(rois_in_feature[0][1] + 0.5)
    roi_start_h = int(rois_in_feature[0][2] + 0.5)
    roi_end_w = int(rois_in_feature[0][3] + 0.5)
    roi_end_h = int(rois_in_feature[0][4] + 0.5)

    roi_width = (roi_end_w - roi_start_w + 1)
    roi_height = (roi_end_h - roi_start_h + 1)

    bin_size_w = roi_width / 7
    bin_size_h = roi_height / 7

    self_result = torch.zeros((7, 7))
    for j in range(7):
        for i in range(7):
            self_result[j][i] = (feat[...,
                                 int(roi_start_h + bin_size_h * j):math.ceil(roi_start_h + bin_size_h * (j + 1)), \
                                 int(roi_start_w + bin_size_w * i):math.ceil(roi_start_w + bin_size_w * (i + 1))][0][ \
                                     0].max().item())
    print(self_result)
    pass


class RoIPoolModel(torch.nn.Module):
    def __init__(self):
        super(RoIPoolModel, self).__init__()
        self.roi_pool = torchvision.ops.RoIPool((7, 7), 1/16)

    def forward(self, feat, rois):
        r = self.roi_pool(feat, rois)
        return r
    pass


def inference(feat, rois, model):
    output = model(feat, rois)
    print(output)
    pass


if __name__ == '__main__':
    N = 1
    C = 1
    H = 50
    W = 50
    feature_maps = torch.randn(N, C, H, W)
    rois_in_feature = torch.Tensor([0, 10, 20, 30, 40])
    rois_in_feature = rois_in_feature.unsqueeze(0)
    rois_in_image = rois_in_feature*16
    self_roi_pool(feature_maps, rois_in_feature)

    net = RoIPoolModel()
    inference(feature_maps, rois_in_image, net)
    # model_name = './roiPoolModel.onnx'
    # torch.onnx.export(model=net, args=(feature_maps, rois_in_image), f=model_name,
    #                   input_names=['feat', 'rois'], output_names=['output'])

    pass