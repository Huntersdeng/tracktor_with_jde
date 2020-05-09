import argparse
import os
import os.path as osp
import json
import yaml
import warnings
from utils.utils import mkdir_if_missing, init_seeds
from tqdm import tqdm
import torch
from torchvision.transforms import transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset
from utils.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from flowTracker import flowTracker

warnings.filterwarnings("ignore")
def train(
        save_path,
        save_every,
        img_size,
        resume,
        epochs,
        opt=None
):
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
    model_name = 'flowNet' 
    weights_path = osp.join(save_path, model_name)
    loss_log_path = osp.join(weights_path, 'loss.json')
    mkdir_if_missing(weights_path)
    cfg = {}
    cfg['lr'] = opt.lr
    cfg['height'] = img_size[1]
    cfg['width'] = img_size[0]
    
    if resume:
        latest_resume = osp.join(weights_path, 'latest.pt')

    torch.backends.cudnn.benchmark = True
    root = '/home/hunter/Document/torch'
    # root = '/data/dgw'

    #paths = {'CT':'./data/detect/CT_train.txt', 
    #         'ETH':'./data/detect/ETH.txt', 'M16':'./data/detect/MOT16_train.txt', 
    #         'PRW':'./data/detect/PRW_train.txt', 'CP':'./data/detect/cp_train.txt'}
    paths_trainset =  {'02':'./data/track/train+val/MOT16-02.txt',
                       '04':'./data/track/train+val/MOT16-04.txt',
                       '05':'./data/track/train+val/MOT16-05.txt',
                       '09':'./data/track/train+val/MOT16-09.txt',
                       '10':'./data/track/train+val/MOT16-10.txt',
                       '11':'./data/track/train+val/MOT16-11.txt',
                       '13':'./data/track/train+val/MOT16-13.txt'}
    transforms = T.Compose([T.ToTensor()])
    trainset = JointDataset(root=root, paths=paths_trainset, img_size=img_size, augment=False, transforms=transforms)

    dataloader_trainset = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    model = flowTracker(img_size)
    model.train()
    # model.cuda().train()

    start_epoch = 0

    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=5e-4)
    after_scheduler = StepLR(optimizer, 10, 0.1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=after_scheduler)

    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        print(model.load_state_dict(checkpoint['model'],strict=False))
        
        start_epoch = checkpoint['epoch'] + 1

        del checkpoint  # current, saved
        
    else:
        with open(osp.join(weights_path,'model.yaml'), 'w+') as f:
            yaml.dump(cfg, f)
        
    for epoch in range(epochs):
        epoch = epoch + start_epoch
        print('lr: ', optimizer.param_groups[0]['lr'])
        scheduler.step(epoch)
        loss_epoch_log = 0
        for i, (imgs, labels, _, _, targets_len) in enumerate(tqdm(dataloader_trainset)):
            # imgs = imgs.cuda()
            # labels = labels.cuda()
            imgs = torch.cat((imgs[0], imgs[1]), dim=0).unsqueeze(dim=0)
            boxes, target = labels[0][:int(targets_len[0][0])], labels[1][:int(targets_len[1][0])]
            loss = model(imgs, boxes, target)
            loss.backward()

        ## print and log the loss

            loss_epoch_log += loss
        
        loss_epoch_log = loss_epoch_log/i
        print("loss in epoch %d: "%(epoch))
        print(loss_epoch_log)
                

        checkpoint = {'epoch': epoch,
                      'model': model.state_dict()
                    }
        latest = osp.join(weights_path, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            torch.save(checkpoint, osp.join(weights_path, "weights_epoch_" + str(epoch) + ".pt"))
        with open(loss_log_path, 'a+') as f:
            f.write('epoch:'+str(epoch)+'\n')
            json.dump(loss_epoch_log, f) 
            f.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--save-path', type=str, default='../',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                                '--resume)')
    parser.add_argument('--save-model-after', type=int, default=5,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--img-size', type=int, default=(768,448), nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--lr', type=float, default=1e-4, help='init lr')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    opt = parser.parse_args()

    init_seeds()

    train(
        save_path=opt.save_path,
        save_every=opt.save_model_after,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        opt=opt
    )
