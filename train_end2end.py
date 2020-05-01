import argparse
import os
import os.path as osp
import json
import yaml
from utils.utils import mkdir_if_missing, init_seeds
from test import test, test_emb
from tqdm import tqdm
import torch
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset, letterbox, random_affine
from utils.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from model import Jde_RCNN

def train(
        save_path,
        save_every,
        train_reid,
        img_size,
        resume,
        epochs,
        batch_size,
        accumulated_batches,
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
    paths_trainset =  {'02':'./data/track/train/MOT16-02.txt',
                       '04':'./data/track/train/MOT16-04.txt',
                       '05':'./data/track/train/MOT16-05.txt',
                       '09':'./data/track/train/MOT16-09.txt',
                       '10':'./data/track/train/MOT16-10.txt',
                       '11':'./data/track/train/MOT16-11.txt',
                       '13':'./data/track/train/MOT16-13.txt',
                       'CT':'./data/detect/CT_train.txt', 
                       'ETH':'./data/detect/ETH.txt',
                       'PRW':'./data/detect/PRW_train.txt', 
                       'CP':'./data/detect/cp_train.txt'}
    paths_valset =    {'02':'./data/track/val/MOT16-02.txt',
                       '04':'./data/track/val/MOT16-04.txt',
                       '05':'./data/track/val/MOT16-05.txt',
                       '09':'./data/track/val/MOT16-09.txt',
                       '10':'./data/track/val/MOT16-10.txt',
                       '11':'./data/track/val/MOT16-11.txt',
                       '13':'./data/track/val/MOT16-13.txt',
                       'CP':'./data/detect/cp_val.txt',
                       'PRW':'./data/detect/PRW_val.txt',
                       'CT':'./data/detect/CT_val.txt'}
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

    model = Jde_RCNN(backbone, num_ID=trainset.nID, min_size=img_size[1], max_size=img_size[0], version=opt.model_version, len_embeddings=256)
    model.cuda().train()

    # model = torch.nn.DataParallel(model)
    start_epoch_det = 0
    start_epoch_reid = 0
    layer = ['roi_heads.embed_head.fc8.weight',
             'roi_heads.embed_head.fc8.bias',
             'roi_heads.embed_head.fc9.weight,'
             'roi_heads.embed_head.fc9.bias,'
             'roi_heads.embed_extractor.extract_embedding.weight,'
             'roi_heads.embed_extractor.extract_embedding.bias,'
             'roi_heads.identifier.weight,'
             'roi_heads.identifier.bias']
    if not train_reid:
        for name, p in model.roi_heads.named_parameters():
            #print(name)
            if name in layer:
                p.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=5e-4)
        after_scheduler = StepLR(optimizer, 10, 0.1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=after_scheduler)
    else:
        
        for name, p in model.named_parameters():
            if name not in layer:
                p.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, 10, 0.1)

    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        print(model.load_state_dict(checkpoint['model'],strict=False))
        
        start_epoch_det = checkpoint['epoch_det'] + 1
        start_epoch_reid = checkpoint['epoch_reid'] + 1

        del checkpoint  # current, saved
        
    else:
        with open(osp.join(weights_path,'model.yaml'), 'w+') as f:
            yaml.dump(cfg, f)
        
    for epoch in range(epochs):
        model.cuda().eval()
        with torch.no_grad():
            if train_reid:
                test_emb(model, dataloader_valset, print_interval=50)[-1]
                scheduler.step(epoch+start_epoch_reid)
            else:
                test(model, dataloader_valset, conf_thres=0.9, print_interval=50)
                
                scheduler.step(epoch+start_epoch_det)
            print(scheduler.get_lr())

        model.cuda().train()
        print('lr: ', optimizer.param_groups[0]['lr'])
        loss_epoch_log = dict(loss_total=0, loss_classifier=0, loss_box_reg=0, loss_reid=0, loss_objectness=0, loss_rpn_box_reg=0)
        for i, (imgs, labels, _, _, targets_len) in enumerate(tqdm(dataloader_trainset)):
            targets = []
            imgs = imgs.cuda()
            labels = labels.cuda()
            flag = False
            for target_len, label in zip(targets_len.view(-1,), labels):
                ## convert the input to demanded format
                target = {}
                if target_len==0:
                    flag = True
                if np.all(label[0:int(target_len), 1]==-1):
                    flag = True
                target['boxes'] = label[0:int(target_len), 2:6]
                target['ids'] = (label[0:int(target_len), 1]).long()
                target['labels'] = torch.ones_like(target['ids'])
                targets.append(target)
            if flag:
                continue
            losses = model(imgs, targets)
            if not train_reid:
                loss = losses['loss_classifier'] + losses['loss_box_reg'] + losses['loss_objectness'] + losses['loss_rpn_box_reg']
            else:
                loss = losses['loss_reid']
            loss.backward()

            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader_trainset) - 1):
                optimizer.step()
                optimizer.zero_grad()
        ## print and log the loss

            for key, val in losses.items():
                loss_epoch_log[key] = float(val) + loss_epoch_log[key]
        
        for key, val in loss_epoch_log.items():
            loss_epoch_log[key] =loss_epoch_log[key]/i
        print("loss in epoch %d: "%(epoch))
        print(loss_epoch_log)
                
        if not train_reid:
            epoch_det = epoch + start_epoch_det
            epoch_reid = start_epoch_reid
        else:
            epoch_det = start_epoch_det
            epoch_reid = epoch + start_epoch_reid

        checkpoint = {'epoch_det': epoch_det,
                      'epoch_reid': epoch_reid,
                      'model': model.state_dict()
                    }
        latest = osp.join(weights_path, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            torch.save(checkpoint, osp.join(weights_path, "weights_epoch_" + str(epoch_det) + '_' + str(epoch_reid) + ".pt"))
        with open(loss_log_path, 'a+') as f:
            f.write('epoch_det:'+str(epoch_det)+',epoch_reid:'+str(epoch_reid)+'\n')
            json.dump(loss_epoch_log, f) 
            f.write('\n')



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
        train_reid=opt.train_reid,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt
    )

