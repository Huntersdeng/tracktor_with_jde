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
pos =  torch.FloatTensor([[ 258.9648,  260.0791,  323.6870,  419.7425],
        [ 777.3400,  244.6645,  876.6724,  455.2278],
        [ 339.2203,  259.4285,  397.8312,  412.5075],
        [ 593.5024,  254.5205,  617.7199,  317.1711],
        [ 832.5743,  245.2603,  932.6730,  453.1967],
        [ 544.2612,  253.1349,  569.4096,  320.2453],
        [ 354.4627,  249.7224,  363.8081,  270.0044],
        [ 307.5739,  262.1659,  321.3314,  298.9787],
        [ 357.8457,  253.7306,  369.3017,  281.4169],
        [ 391.7324,  265.6148,  406.7554,  307.5581],
        [ 325.3136,  257.8607,  336.9111,  285.9368]])

well = torch.FloatTensor([[,456,280,79,239],
[321,176,59,180],
[1522,103,59,180],
[1837,271,48,146],
[1488,784,159,479],
[820,221,55,168],
[285,285,59,180],
[1684,111,51,157],
[285,230,90,274],
[1445,81,41,127],
[542,187,97,294],
[1684,483,97,294]])


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
print(model.predict_boxes(torch.cat(dets,pos)))
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
