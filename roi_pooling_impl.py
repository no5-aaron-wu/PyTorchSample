import torchvision
from tensorflow.keras.preprocessing.image import load_img
import torchvision.transforms as T
import torch
# from torchvision.ops import roi_pool
import math

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

img = load_img('2007_000032.jpg', target_size=(800, 800))
transform = T.Compose([T.ToTensor()])
img = transform(img)


images = [img]
targets=None
original_image_sizes = [img.shape[-2:] for img in images]
images, targets = model.transform(images, targets)

# 特征提取
features = model.backbone(images.tensors)
# 提取proposals
proposals, proposal_losses = model.rpn(images, features, targets)
print(proposals[0].size())
# proposals的大小是1000，我们随机选取一个进行计算
test_proposal = proposals[0][76]
image_shape = images.image_sizes
third_level_feature = features['2']
spatial_scale = third_level_feature.size()[2] / image_shape[0][0]

# 特征图上的roi
rois_in_feature = test_proposal*spatial_scale
rois_in_feature = torch.cat((torch.zeros(1), rois_in_feature), dim=0)
rois_in_feature = rois_in_feature.unsqueeze(0)

rois_in_image = torch.cat((torch.zeros(1), test_proposal), dim=0)
rois_in_image = rois_in_image.unsqueeze(0)

# torchvision 计算
roi_pool = torchvision.ops.RoIPool((7, 7), spatial_scale)
tv_result = roi_pool(third_level_feature, rois_in_image)
print(tv_result.shape)
print(tv_result[0][0])

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
        self_result[j][i] = (third_level_feature[..., int(roi_start_h+bin_size_h*j):math.ceil(roi_start_h+bin_size_h*(j+1)), \
                       int(roi_start_w+bin_size_w*i):math.ceil(roi_start_w+bin_size_w*(i+1))][0][0].max().item())
print(self_result)

a = 0
