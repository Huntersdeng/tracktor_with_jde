import os
import warnings

import torch
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset, letterbox, random_affine
from jde_rcnn import Jde_RCNN

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

root = '/home/hunter/Document/torch'
# root = '/data/dgw'

paths = {'CT':'./data/CalTech.txt', 'CUHK':'./data/CUHK.txt', 
         'ETH':'./data/ETH.txt', 'M16':'/data/MOT16_train.txt', 
         'PRW':'./data/PRW.txt', 'CP':'./data/cp_train.txt'}
transforms = T.Compose([T.ToTensor(),
                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])])
trainset = JointDataset(root=root, paths={'ETH':'/home/hunter/Document/torch/myproject/data/ETH.txt'}, img_size=(576,320), augment=True, transforms=transforms)
# trainset = LoadImagesAndLabels(root, paths['ETH'], img_size=(576,320), augment=True, transforms=transforms)

batch_size = 4
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

backbone = resnet_fpn_backbone('resnet50', True)
backbone.out_channels = 256

model = Jde_RCNN(backbone, num_ID=trainset.nID)

for i, (imgs, labels, _, _, targets_len) in enumerate(dataloader):
    targets = []
    for target_len, label in zip(np.squeeze(targets_len), labels):
        target = {}
        target['boxes'] = label[0:int(target_len), 2:6]
        target['labels'] = (label[0:int(target_len), 1]).long()
        targets.append(target)
    losses = model(imgs, targets)
    print(losses)
    break
