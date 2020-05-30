import os
import time
from os import path as osp
import yaml
from tqdm import tqdm
import argparse

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
from utils.datasets import LoadImagesAndDets, letterbox
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--with-labels', action='store_true', help='for valset')
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
root = '/data/dgw/'
# root = '..'
output_dir = '../output'


print("Initializing object detector.")
with open('./cfg/test_tracktor.yaml', 'r') as f:
    tracktor = yaml.load(f,Loader=yaml.FullLoader)['tracktor']

img_size = (tracktor['width'], tracktor['height'])
with_dets = tracktor['tracker']['public_detections']

##########################
# Initialize the modules #
##########################

# object detection
backbone = resnet_fpn_backbone(tracktor['backbone'], True)
backbone.out_channels = 256
obj_detect = Jde_RCNN(backbone, num_ID=tracktor['num_ID'], min_size=img_size[1], max_size=img_size[0], version=tracktor['version'], len_embeddings=tracktor['len_embed'])
checkpoint = torch.load(tracktor['weights'], map_location='cpu')['model']
# if tracktor['version']=='v2':
#     checkpoint['roi_heads.embed_extractor.extract_embedding.weight'] = checkpoint['roi_heads.box_predictor.extract_embedding.weight']
#     checkpoint['roi_heads.embed_extractor.extract_embedding.bias'] = checkpoint['roi_heads.box_predictor.extract_embedding.bias']
print(obj_detect.load_state_dict(checkpoint, strict=False))

obj_detect.cuda().eval()

tracker = Tracker(obj_detect, tracktor['tracker'])

transforms = T.Compose([T.ToTensor()])

time_total = 0
time_step = {'load':0.0, 'det':0.0,'regress':0.0,'motion':0.0,'reid':0.0,'track':0.0, 'step':0.0}
num_frames = 0
mot_accums = []

for seq_path in os.listdir(tracktor['dataset']):
    tracker.reset()
    

    print(f"Tracking: {seq_path}")
    sequence = LoadImagesAndDets(root, osp.join(tracktor['dataset'], seq_path), img_size, opt.with_labels, with_dets)
    data_loader = DataLoader(sequence, batch_size=2, shuffle=False)

    start = time.time()
    seq = []
    for i, (_, frame, _, dets, labels) in enumerate(tqdm(data_loader)):
        
        blob = {'img':frame[0].cuda(), 'dets':dets[0,:,2:6].cuda()} if with_dets else {'img':frame.cuda(), 'dets':None}
        # blob = {'img':frame, 'dets':dets[0,:,2:6]}
        with torch.no_grad():
            tracker.step(blob)
        num_frames += 1
        
        if opt.with_labels:
            gt = {}
            for label in labels[0]:
                gt[label[1]] = label[2:6]
            seq.append({'gt':gt})
    results = tracker.get_results()
    time_total += tracker.time['step']
    for key in time_step.keys():
        time_step[key] += tracker.time[key]

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq_path}: {tracker.time['step'] :.1f} s.")
    print('Runtime for different steps: ', tracker.time)
    print('nums of boxes: ', tracker.boxes)

    if tracktor['interpolate']:
        results = interpolate(results)

    if opt.with_labels:
        mot_accums.append(get_mot_accum(results, seq))

    print(f"Writing predictions to: {output_dir}")
    write_results(seq_path.rstrip('.txt'), results, output_dir, img_size)

    if tracktor['write_images']:
        plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], seq_path).rstrip('.txt'), img_size)

print(f"Tracking runtime for all sequences (without evaluation or image writing): "
            f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
print('Runtime for different steps: ', time_step)
if opt.with_labels:
    evaluate_mot_accums(mot_accums, [str(s) for s in os.listdir(tracktor['dataset'])], generate_overall=True)
