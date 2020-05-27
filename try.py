import os
import numpy as np
import torch
import cv2
from torchvision.transforms import transforms as T
import time

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model import Jde_RCNN
from utils.datasets import letterbox

backbone = resnet_fpn_backbone('resnet50', True)
backbone.out_channels = 256
transforms = T.Compose([T.ToTensor()])
model = Jde_RCNN(backbone, num_ID=1443, min_size=630, max_size=1120, len_embeddings=1024)
model.load_state_dict(torch.load('../weights/training/all/resnet50_img_size1120_630/latest.pt', map_location='cpu')['model'])
model.cuda().eval()
dets = torch.FloatTensor([[ 600.4134,  259.4911,  630.3846,  329.3570]])
pos =  torch.FloatTensor([[ 258.9648,  260.0791,  323.6870,  419.7425]])
well = torch.FloatTensor([[ 777.3400,  244.6645,  876.6724,  455.2278],
                          [ 339.2203,  259.4285,  397.8312,  412.5075]])


start = time.time()
img = cv2.imread('/data/dgw/dataset/MOT16/train/MOT16-02/images/000001.jpg')
img, _, _, _ =letterbox(img, height=630, width=1120)
img = np.ascontiguousarray(img[ :, :, ::-1])
img = transforms(img).unsqueeze(0)
print('Runtime for pre img1: ', time.time()-start)

start = time.time()
model.load_image(img)
print('Runtime for load img1: ', time.time()-start)

start = time.time()
print(model.predict_boxes(dets))
print('Runtime: ', time.time()-start)

start = time.time()
print(model.predict_boxes(pos))
print('Runtime: ', time.time()-start)

start = time.time()
print(model.predict_boxes(well))
print('Runtime: ', time.time()-start)

start = time.time()
img = cv2.imread('/data/dgw/dataset/MOT16/train/MOT16-02/images/000002.jpg')
img, _, _, _ =letterbox(img, height=630, width=1120)
img = np.ascontiguousarray(img[ :, :, ::-1])
img = transforms(img).unsqueeze(0)
print('Runtime for pre img2: ', time.time()-start)

start = time.time()
model.load_image(img)
print('Runtime for laod img2: ', time.time()-start)

start = time.time()
print(model.predict_boxes(torch.cat((dets,pos))))
print('Runtime: ', time.time()-start)

# start = time.time()
# print(model.predict_boxes(pos))
# print('Runtime: ', time.time()-start)

# start = time.time()
# print(model.predict_boxes(well))
# print('Runtime: ', time.time()-start)

start = time.time()
img = cv2.imread('/data/dgw/dataset/MOT16/train/MOT16-02/images/000003.jpg')
img, _, _, _ =letterbox(img, height=630, width=1120)
img = np.ascontiguousarray(img[ :, :, ::-1])
img = transforms(img).unsqueeze(0)
print('Runtime for pre img3: ', time.time()-start)

start = time.time()
model.load_image(img)
print('Runtime for laod img3: ', time.time()-start)

# start = time.time()
# print(model.predict_boxes(dets))
# print('Runtime: ', time.time()-start)

# start = time.time()
# print(model.predict_boxes(pos))
# print('Runtime: ', time.time()-start)

start = time.time()
print(model.predict_boxes(well))
print('Runtime: ', time.time()-start)
