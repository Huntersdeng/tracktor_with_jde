from utils.FlowNetS import FlowNet2S
from torchsummary import summary
import torch
from model import Jde_RCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np
import argparse
from torchvision.transforms import transforms as T
from utils.datasets import LoadImagesAndLabels_2
from utils.utils import flow_to_image
import cv2


model = FlowNet2S(1)
# summary(model, (6,448,768))
# print(model.state_dict().keys())
model.load_state_dict(torch.load('../weights/flownets_from_caffe.pth.tar.pth'))

root = '/home/hunter/Document/torch'
# root = '/data/dgw'

paths_trainset =  './data/flow/MOT16.txt'
# transforms = T.Compose([T.ToTensor()])
# trainset = LoadImagesAndLabels_2(root=root, path=paths_trainset, img_size=(768,448), augment=False, transforms=transforms)

# dataloader_trainset = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
# model.eval()
summary(model, (3,2,448,768))
# for i, (imgs, labels, img_path, _) in enumerate(dataloader_trainset):
#     break
# cv2.imwrite('test.jpg',imgs[0][0].numpy()[:, :, ::-1])
# print(img_path)
# imgs = torch.cat(imgs, dim=0)
# imgs = imgs.permute(1, 0, 2, 3).unsqueeze(0)
# img1 = cv2.resize(cv2.imread('frame_0008.png'), (1024,384))
# img2 = cv2.resize(cv2.imread('frame_0009.png'), (1024,384))
# imgs = [img1, img2]

# imgs = np.array(imgs).transpose(3, 0, 1, 2)
# imgs = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(0)
# out = model(imgs)
# flow = out[0].detach().numpy().transpose(1,2,0)
# flow = flow_to_image(flow)
# cv2.imwrite('flow.jpg',flow)