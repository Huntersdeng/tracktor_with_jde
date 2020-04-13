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
from jde_rcnn import Jde_RCNN

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='2'

def train(
        weights_from,
        weights_to,
        save_every,
        train_rpn_stage,
        img_size=(640,480),
        resume=False,
        epochs=25,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None
):

    model_name = opt.backbone_name + '_img_size' + str(img_size[0]) + '_' + str(img_size[1]) 
    weights_to = osp.join(weights_to, model_name)
    loss_log_path = './log/loss_' + model_name + '.json'
    mkdir_if_missing(weights_to)
    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True
    # root = '/home/hunter/Document/torch'
    root = '/data/dgw'

    paths = {'CT':'./data/CalTech.txt', 
             'ETH':'./data/ETH.txt', 'M16':'./data/MOT16_train.txt', 
             'PRW':'./data/PRW.txt', 'CP':'./data/cp_train.txt'}
    #paths = {'M16':'./data/cp_train.txt'}
    transforms = T.Compose([T.ToTensor()])
    trainset = JointDataset(root=root, paths=paths, img_size=img_size, augment=False, transforms=transforms)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    backbone = resnet_fpn_backbone(opt.backbone_name, True)
    backbone.out_channels = 256

    model = Jde_RCNN(backbone, num_ID=trainset.nID)
    # model = torch.nn.DataParallel(model)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()

        # Set optimizer
        optimizer_rpn = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)
        optimizer_roi = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer_rpn'] is not None:
            optimizer_rpn.load_state_dict(checkpoint['optimizer_rpn'])
        if checkpoint['optimizer_roi'] is not None:
            optimizer_roi.load_state_dict(checkpoint['optimizer_roi'])            

        del checkpoint  # current, saved
        #with open(loss_log_path, 'r') as file:
        #   loss_log = json.load(file)
        loss_log = []
    else:
        model.cuda().train()

        # Set optimizer
        optimizer_roi = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                    weight_decay=5e-4)
        optimizer_rpn = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                    weight_decay=5e-4)

        loss_log = []

    

    for epoch in range(epochs):
        epoch += start_epoch
        if epoch>=train_rpn_stage:
            for i, (name, p) in enumerate(model.backbone.named_parameters()):
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
            if epoch < train_rpn_stage:
                loss = losses['loss_objectness'] + losses['loss_rpn_box_reg']
                loss.backward()
                if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                    optimizer_rpn.step()
                    optimizer_rpn.zero_grad()
            else:
                # loss = (losses['loss_objectness'] + losses['loss_rpn_box_reg'])*0.2
                # loss.backward(retain_graph=True)
                losses['loss_total'].backward()
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
        loss_log.append(loss_epoch_log)
        

        checkpoint = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer_rpn': optimizer_rpn.state_dict(),
                      'optimizer_roi': optimizer_roi.state_dict()}

        latest = osp.join(weights_to, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            # making the checkpoint lite
            checkpoint["optimizer"] = []
            torch.save(checkpoint, osp.join(weights_to, "weights_epoch_" + str(epoch) + ".pt"))
    with open(loss_log_path, 'w+') as f:
        json.dump(loss_log, f) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--weights-from', type=str, default='../weights/',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                                '--resume)')
    parser.add_argument('--weights-to', type=str, default='../weights/',
                        help='Store the trained weights after resuming training session. It will create a new folder '
                                'with timestamp in the given path')
    parser.add_argument('--save-model-after', type=int, default=5,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--train-rpn-stage', type=int, default=10, help='for training rpn')
    parser.add_argument('--img-size', type=int, default=(960,720), nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--lr', type=float, default=5e-4, help='init lr')
    parser.add_argument('--backbone-name', type=str, default='resnet101', help='backbone name')
    opt = parser.parse_args()

    init_seeds()

    train(
        weights_from=opt.weights_from,
        weights_to=opt.weights_to,
        save_every=opt.save_model_after,
        train_rpn_stage=opt.train_rpn_stage,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt
    )

