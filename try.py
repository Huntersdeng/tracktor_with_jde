from flowTracker import flowTracker, match
from torchsummary import summary
import torch
from model import Jde_RCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np

# model = flowTracker()
# model.eval()
# imgs = torch.randn((1,6,448,768))
# box = torch.tensor([[1,2,3,4],[3,4,5,6]])
# out = model(imgs, box)
# print(out)
# backbone = resnet_fpn_backbone('resnet50', True)
# backbone.out_channels = 256

# model = Jde_RCNN(backbone, num_ID=100)

# imgs = [torch.randn((3,448,768))]
# model.eval()
# model(imgs)
label_path1 = '../dataset/MOT16/train/MOT16-02/labels_with_ids/000001.txt'
label_path2 = '../dataset/MOT16/train/MOT16-02/labels_with_ids/000002.txt'

label1 = torch.from_numpy(np.loadtxt(label_path1, delimiter=','))
label2 = torch.from_numpy(np.loadtxt(label_path2, delimiter=','))

out1, out2 = match(label1, label2)
print(out1)
print(out2)
