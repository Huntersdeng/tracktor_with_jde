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
from utils.datasets import LoadImages

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='resnet50', help='type of backbone')
parser.add_argument('--img-size', type=int, default=(800,450), nargs='+', help='pixels')
parser.add_argument('--with-labels', action='store_true', help='for valset')
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
root = '/data/dgw/'
# root = '..'
output_dir = '../output'
width = str(opt.img_size[0])
height = str(opt.img_size[1])


print("Initializing object detector.")
with open('./cfg/tracktor_'+opt.backbone+'_'+width+'_'+height+'.yaml', 'r') as f:
    tracktor = yaml.load(f,Loader=yaml.FullLoader)['tracktor']

img_size = (tracktor['width'], tracktor['height'])
with_dets = tracktor['tracker']['public_detections']

##########################
# Initialize the modules #
##########################

# object detection
backbone = resnet_fpn_backbone(tracktor['backbone'], True)
backbone.out_channels = 256
obj_detect = Jde_RCNN(backbone, num_ID=tracktor['num_ID'], min_size=img_size[1], max_size=img_size[0], version=tracktor['version'])
checkpoint = torch.load(tracktor['weights'], map_location='cpu')['model']
# if tracktor['version']=='v2':
#     checkpoint['roi_heads.embed_extractor.extract_embedding.weight'] = checkpoint['roi_heads.box_predictor.extract_embedding.weight']
#     checkpoint['roi_heads.embed_extractor.extract_embedding.bias'] = checkpoint['roi_heads.box_predictor.extract_embedding.bias']
print(obj_detect.load_state_dict(checkpoint, strict=False))

obj_detect.eval()
obj_detect.cuda()

tracker = Tracker(obj_detect, tracktor['tracker'])

transforms = T.Compose([T.ToTensor()])

time_total = 0
num_frames = 0
mot_accums = []

for seq_path in os.listdir(tracktor['dataset']):
    tracker.reset()

    start = time.time()

    print(f"Tracking: {seq_path}")
    sequence = LoadImages(osp.join(tracktor['dataset'], seq_path+'/images'), img_size, opt.with_labels, with_dets)
    data_loader = DataLoader(sequence, batch_size=1, shuffle=False)

    seq = []
    for i, (_, frame, _, dets, labels) in enumerate(tqdm(data_loader)):
        blob = {'img':frame.cuda(), 'dets':dets[0,:,2:6]}
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
    time_total += time.time() - start

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq_path}: {time.time() - start :.1f} s.")

    if tracktor['interpolate']:
        results = interpolate(results)

    if opt.with_labels:
        mot_accums.append(get_mot_accum(results, seq))

    print(f"Writing predictions to: {output_dir}")
    write_results(seq_path.rstrip('.txt'), results, output_dir)

    if tracktor['write_images']:
        plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], seq_path).rstrip('.txt'))

print(f"Tracking runtime for all sequences (without evaluation or image writing): "
            f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
if opt.with_labels:
    evaluate_mot_accums(mot_accums, [str(s) for s in os.listdir(tracktor['dataset'])], generate_overall=True)
    evaluate_mot_accums(mot_accums, ['a'], generate_overall=True)
