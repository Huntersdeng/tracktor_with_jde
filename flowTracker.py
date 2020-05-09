import torch
import torch.nn as nn
from utils.FlowNetS import FlowNetS
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
import numpy as np
from torchvision.ops.boxes import clip_boxes_to_image

class flowTracker(nn.Module):
    def __init__(self, img_size):
        super(flowTracker, self).__init__()
        self.flownet = FlowNetS(input_channels=6)
        self.box_roi_pool = torch.nn.AdaptiveMaxPool2d((7,7), return_indices=False)
        self.fc = nn.Linear(7**2*2, 4)
        self.img_size = img_size
    
    def forward(self, x, boxes, target=None):
        feature = self.flownet(x)
        
        if self.training:
            boxes, target = match(boxes, target)

        box_feature = [self.box_roi_pool(feature[:,:,int(box[1]):int(box[3]),int(box[0]):int(box[2])]) for box in boxes]
        box_feature = torch.cat(box_feature, dim=0)
        box_feature = box_feature.flatten(start_dim=1)
        deltaB = F.relu(self.fc(box_feature))

        if self.training:
            return F.smooth_l1_loss(deltaB, target-boxes)

        return deltaB + boxes

def match(boxes, target):
    
    m, n = boxes.shape[0], target.shape[0]
    idx = torch.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if boxes[i,1]==target[j,1]:
                idx[i,j] = 1
                break
    boxes = boxes[idx.sum(dim=1, dtype=torch.uint8)]
    target = target[idx.sum(dim=0, dtype=torch.uint8)]
    boxes = boxes[torch.argsort(boxes[:,1])]
    target = target[torch.argsort(target[:,1])]
    boxes = clip_boxes_to_image(boxes[:,2:6], self.img_size)
    target = clip_boxes_to_image(target[:,2:6], self.img_size)
    return boxes, target



