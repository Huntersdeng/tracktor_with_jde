import argparse
import os.path as osp
import warnings
import json
import time
from time import gmtime, strftime
from utils.utils import *

import torch
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset, letterbox, random_affine
from model import Jde_RCNN

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='1'

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

    model_name = opt.backbone_name + '_img_size' + str(img_size[0]) + '_' + str(img_size[1]) 
    weights_path = osp.join(save_path, model_name)
    loss_log_path = osp.join(weights_path, 'loss.json')
    mkdir_if_missing(weights_path)
    if resume:
        latest_resume = osp.join(weights_path, 'latest.pt')

    torch.backends.cudnn.benchmark = True
    # root = '/home/hunter/Document/torch'
    root = '/data/dgw'

    paths = {'CT':'./data/detect/CT_train.txt', 
             'ETH':'./data/detect/ETH.txt', 'M16':'./data/detect/MOT16_train.txt', 
             'PRW':'./data/detect/PRW_train.txt', 'CP':'./data/detect/cp_train.txt'}
    #paths = {'M16':'./data/cp_train.txt'}
    transforms = T.Compose([T.ToTensor()])
    trainset = JointDataset(root=root, paths=paths, img_size=img_size, augment=False, transforms=transforms)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    backbone = resnet_fpn_backbone(opt.backbone_name, True)
    backbone.out_channels = 256

    model = Jde_RCNN(backbone, num_ID=trainset.nID, min_size=img_size[1], max_size=img_size[0])
    # model = torch.nn.DataParallel(model)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()

        # Set optimizer
        optimizer_rpn = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)
        optimizer_roi = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)
        optimizer_reid = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer_rpn'] is not None:
            optimizer_rpn.load_state_dict(checkpoint['optimizer_rpn'])
        if checkpoint['optimizer_roi'] is not None:
            optimizer_roi.load_state_dict(checkpoint['optimizer_roi'])
        if checkpoint['optimizer_reid'] is not None:
            optimizer_reid.load_state_dict(checkpoint['optimizer_roi'])            

        del checkpoint  # current, saved
        
    else:
        model.cuda().train()

        # Set optimizer
        optimizer_roi = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                    weight_decay=5e-4)
        optimizer_rpn = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                    weight_decay=5e-4)
        optimizer_reid = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                    weight_decay=5e-4)


    

    for epoch in range(epochs):
        epoch += start_epoch
        if not train_rpn_stage:
            for i, (name, p) in enumerate(model.backbone.named_parameters()):
                p.requires_grad = False
            for i, (name, p) in enumerate(model.rpn.named_parameters()):
                p.requires_grad = False
        else:
            for i, (name, p) in enumerate(model.roi_heads.named_parameters()):
                p.requires_grad = False
        loss_epoch_log = dict(loss_total=0, loss_classifier=0, loss_box_reg=0, loss_reid=0, loss_objectness=0, loss_rpn_box_reg=0)
        for i, (imgs, labels, imgs_path, _, targets_len) in enumerate(dataloader):
            targets = []
            imgs = imgs.cuda()
            labels = labels.cuda()
            for target_len, label in zip(np.squeeze(targets_len), labels):
                ## convert the input to demanded format
                target = {}
                target['boxes'] = label[0:int(target_len), 2:6]
                target['ids'] = (label[0:int(target_len), 1]).long()
                target['labels'] = torch.ones_like(target['ids'])
                targets.append(target)
            
            losses = model(imgs, targets)

            ## two stages training
            if train_rpn_stage:
                loss = losses['loss_objectness'] + losses['loss_rpn_box_reg']
                loss.backward()
                if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                    optimizer_rpn.step()
                    optimizer_rpn.zero_grad()
            else:
                if train_reid:
                    loss = losses['loss_reid']
                    loss.backward()
                    if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                        optimizer_reid.step()
                        optimizer_reid.zero_grad()
                else:
                    loss = losses['loss_box_reg'] + losses['loss_classifier']
                    loss.backward()
                    if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                        optimizer_roi.step()
                        optimizer_roi.zero_grad()
                

        ## print and log the loss

            for key, val in losses.items():
                loss_epoch_log[key] = float(val) + loss_epoch_log[key]
            
        for key, val in loss_epoch_log.items():
            loss_epoch_log[key] =loss_epoch_log[key]/i
        print("loss in epoch %d: "%(epoch))
        print(loss_epoch_log)
                

        checkpoint = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer_rpn': optimizer_rpn.state_dict(),
                      'optimizer_roi': optimizer_roi.state_dict(),
                      'optimizer_reid': optimizer_reid.state_dict()}

        latest = osp.join(weights_path, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            # making the checkpoint lite
            checkpoint["optimizer_rpn"] = []
            checkpoint["optimizer_roi"] = []
            checkpoint["optimizer_reid"] = []
            torch.save(checkpoint, osp.join(weights_path, "weights_epoch_" + str(epoch) + ".pt"))
        with open(loss_log_path, 'a+') as f:
            json.dump(loss_epoch_log, f) 



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
    parser.add_argument('--lr', type=float, default=1e-3, help='init lr')
    parser.add_argument('--backbone-name', type=str, default='resnet101', help='backbone name')
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

