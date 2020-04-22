import argparse
import os.path as osp
import warnings
import json
import yaml
import time
from time import gmtime, strftime
from utils.utils import *
from test import test, test_emb
from tqdm import tqdm
import torch
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset, letterbox, random_affine
from utils.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from model2 import Jde_RCNN

import cv2
import matplotlib.pyplot as plt
import numpy as np

from engine import train_one_epoch
import utils

def train(
        save_path,
        save_every,
        train_rpn_stage,
        train_reid,
        img_size=(640,480),
        resume=False,
        epochs=25,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None
):
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
    model_name = opt.backbone_name + '_img_size' + str(img_size[0]) + '_' + str(img_size[1]) 
    weights_path = osp.join(save_path, model_name)
    loss_log_path = osp.join(weights_path, 'loss.json')
    mkdir_if_missing(weights_path)
    cfg = {}
    cfg['width'] = img_size[0]
    cfg['height'] = img_size[1]
    cfg['backbone_name'] = opt.backbone_name
    cfg['lr'] = opt.lr
    
    if resume:
        latest_resume = osp.join(weights_path, 'latest.pt')

    torch.backends.cudnn.benchmark = True
    # root = '/home/hunter/Document/torch'
    root = '/data/dgw'

    #paths = {'CT':'./data/detect/CT_train.txt', 
    #         'ETH':'./data/detect/ETH.txt', 'M16':'./data/detect/MOT16_train.txt', 
    #         'PRW':'./data/detect/PRW_train.txt', 'CP':'./data/detect/cp_train.txt'}
    paths_trainset = {'M16':'./data/detect/MOT16_train.txt'}
    paths_valset = {'M16':'./data/detect/MOT16_val.txt'}
    transforms = T.Compose([T.ToTensor()])
    trainset = JointDataset(root=root, paths=paths_trainset, img_size=img_size, augment=True, transforms=transforms)
    valset = JointDataset(root=root, paths=paths_valset, img_size=img_size, augment=False, transforms=transforms)

    dataloader_trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader_valset = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)                                       
    
    cfg['num_ID'] = trainset.nID
    backbone = resnet_fpn_backbone(opt.backbone_name, True)
    backbone.out_channels = 256

    model = Jde_RCNN(backbone, num_ID=trainset.nID, min_size=img_size[1], max_size=img_size[0], version=opt.model_version)
    model.cuda().train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00001,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(latest_resume, map_location='cpu')

    # Load weights to resume from
    print(model.load_state_dict(checkpoint['model'],strict=False))
        
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, dataloader_trainset, device, epoch, print_freq=200)


        if epoch % save_every == 0:
            torch.save(checkpoint, osp.join(weights_path, "weights_epoch_" + str(epoch) + ".pt"))
            test(model, dataloader_valset, print_interval=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--save-path', type=str, default='../',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                                '--resume)')
    parser.add_argument('--save-model-after', type=int, default=5,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--train-rpn-stage', action='store_true', help='for training rpn')
    parser.add_argument('--train-reid', action='store_true', help='for training reid')
    parser.add_argument('--img-size', type=int, default=(960,720), nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--lr', type=float, default=-1.0, help='init lr')
    parser.add_argument('--backbone-name', type=str, default='resnet101', help='backbone name')
    parser.add_argument('--model-version', type=str, default='v1', help='model')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    opt = parser.parse_args()

    init_seeds()

    train(
        save_path=opt.save_path,
        save_every=opt.save_model_after,
        train_rpn_stage=opt.train_rpn_stage,
        train_reid=opt.train_reid,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt
    )


    