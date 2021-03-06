import torch
import torch.nn as nn
from utils.FlowNetS import FlowNet2S
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
import numpy as np
from torchvision.ops.boxes import clip_boxes_to_image

class flowTracker(nn.Module):
    def __init__(self, img_size):
        super(flowTracker, self).__init__()
        self.flownet = FlowNet2S(1)
        self.flownet.load_state_dict(torch.load('../weights/flownets_from_caffe.pth.tar.pth'))
        self.box_roi_pool = torch.nn.AdaptiveMaxPool2d((10,30), return_indices=False)
        self.fc1 = nn.Linear(2*10*30, 1024)
        self.fc2 = nn.Linear(1024, 4)
        self.img_size = img_size
    
    def forward(self, x, boxes, targets=None, img_path=None):
        feature = self.flownet(x)
        
        if self.training:
            boxes, targets = self.match(boxes, targets)
        box_feature = []
        idx = []
        for box in boxes:
            try:
                box_feature.append(self.box_roi_pool(feature[:,:,int(box[3]):int(box[5]),int(box[2]):int(box[4])]))
                idx.append(1)
            except RuntimeError:
                idx.append(0)
        idx = torch.tensor(idx, dtype=torch.uint8)
        targets = targets[idx]
        boxes = boxes[idx]
        if len(box_feature)>0:
            box_feature = torch.cat(box_feature, dim=0)
        else:
            return None
        box_feature = box_feature.flatten(start_dim=1)
        box_feature = F.relu(self.fc1(box_feature))
        deltaB = F.relu(self.fc2(box_feature))

        if self.training:
            return F.smooth_l1_loss(deltaB, targets[:,2:6]-boxes[:,2:6])
        boxes[:,2:6] = deltaB + boxes[:,2:6]
        return boxes

    def match(self, boxes, targets):
        
        m, n = boxes.shape[0], targets.shape[0]
        idx = torch.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if boxes[i,1]==targets[j,1]:
                    idx[i,j] = 1
                    break
        boxes = boxes[idx.sum(dim=1, dtype=torch.uint8)]
        targets = targets[idx.sum(dim=0, dtype=torch.uint8)]
        boxes = boxes[torch.argsort(boxes[:,1])]
        targets = targets[torch.argsort(targets[:,1])]
        boxes[:,2:6] = clip_boxes_to_image(boxes[:,2:6], (self.img_size[1], self.img_size[0]))
        targets[:,2:6] = clip_boxes_to_image(targets[:,2:6], (self.img_size[1], self.img_size[0]))
        return boxes, targets



