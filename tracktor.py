import os
import time
from os import path as osp
import yaml
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model import Jde_RCNN
from tracker import Tracker
from utils.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums, write_results
from utils.datasets import LoadImagesAndLabels

os.environ['CUDA_VISIBLE_DEVICES']='0'
root = '/data/dgw/'
#root = '..'
output_dir = './output'


print("Initializing object detector.")
with open('./cfg/tracktor.yaml', 'r') as f:
    tracktor = yaml.load(f,Loader=yaml.FullLoader)['tracktor']

##########################
# Initialize the modules #
##########################

# object detection
backbone = resnet_fpn_backbone(tracktor['backbone'], True)
backbone.out_channels = 256
obj_detect = Jde_RCNN(backbone, num_ID=tracktor['num_ID'])
print(obj_detect.load_state_dict(torch.load(tracktor['weights'], map_location='cpu')['model'], strict=False))

obj_detect.eval()
obj_detect.cuda()

tracker = Tracker(obj_detect, tracktor['tracker'])
img_size = (tracktor['width'], tracktor['height'])

transforms = T.Compose([T.ToTensor()])

time_total = 0
num_frames = 0
mot_accums = []

for seq_path in os.listdir(tracktor['dataset']):
    tracker.reset()

    start = time.time()

    print(f"Tracking: {seq_path}")
    sequence = LoadImagesAndLabels(root, osp.join(tracktor['dataset'], seq_path), img_size, augment=False, transforms=transforms)
    data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
    seq = []
    for i, (frame, labels, imgs_path, _) in enumerate(tqdm(data_loader)):
        gt = {}
        for label in labels[0]:
            gt[label[1]] = label[2:6]
        seq.append({'gt':gt})
        blob = {'img':frame.cuda()}
        with torch.no_grad():
            tracker.step(blob)
        num_frames += 1
    results = tracker.get_results()

    time_total += time.time() - start

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq_path}: {time.time() - start :.1f} s.")

    if tracktor['interpolate']:
        results = interpolate(results)

    mot_accums.append(get_mot_accum(results, seq))

    print(f"Writing predictions to: {output_dir}")
    write_results(seq_path.rstrip('.txt'), results, output_dir)

    if tracktor['write_images']:
        plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

print(f"Tracking runtime for all sequences (without evaluation or image writing): "
            f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
if mot_accums:
    evaluate_mot_accums(mot_accums, [str(s) for s in os.listdir(tracktor['dataset'])], generate_overall=True)
