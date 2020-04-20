import argparse
import json
import time
from pathlib import Path
from functools import reduce
import numpy as np
import os
import yaml

from sklearn import metrics
from scipy import interpolate
import torch
import torch.nn.functional as F
from model import Jde_RCNN
from utils.utils import xyxy2xywh, non_max_suppression, ap_per_class, bbox_iou
from torchvision.transforms import transforms as T
from utils.datasets import LoadImagesAndLabels, JointDataset, collate_fn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def test(
        model,
        dataloader,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
):
    
    
    nC = 1
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        out = model(imgs.cuda())
        # out = model(imgs)
        output = []
        for i,o in enumerate(out):
            boxes = xyxy2xywh(o['boxes']).cpu()
            scores = o['scores'].cpu().view(-1,1)
            labels = o['labels'].cpu().view(-1,1).float()
            output.append(torch.Tensor(torch.cat((boxes,scores,scores,labels),dim=1)))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        for i, o in enumerate(output):
            if o is not None:
                output[i] = o[:, :6]

        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
        for si, (labels, detections) in enumerate(zip(targets, output)):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]


            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = torch.zeros_like(labels[:, 0])
                target_boxes = labels[:, 2:6]

                detected = []
                for *pred_bbox, conf, obj_conf  in detections:
                    obj_pred = 0
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=detections[:, 4],
                                              pred_cls=np.zeros_like(detections[:, 5]), # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / ( AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / ( AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval==0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P


def test_emb(
            model,
            dataloader,
            iou_thres=0.5,
            conf_thres=0.3,
            nms_thres=0.45,
            print_interval=40,
):
    embedding, id_labels = [], []
    print('Extracting pedestrain features...')
    for batch_i, (imgs, labels, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        boxes = []
        imgs = imgs.cuda()
        labels = labels.cuda()
        ids = []
        for target_len, label in zip(np.squeeze(targets_len), labels):
            ## convert the input to demanded format
            ids += list(label[0:int(target_len), 1])
            boxes.append(label[0:int(target_len), 2:6])
        model.load_image(imgs)
        embeddings = model.get_embedding(boxes)
        for id_label, feat in zip(ids, embeddings):
            embedding.append(feat.view(1,-1))
            id_labels.append(id_label)
        
        if batch_i % print_interval==0:
            print('Extracting {}/{}, # of instances {}, time {:.2f} sec.'.format(batch_i, len(dataloader), len(id_labels), time.time() - t))

    print('Computing pairwise similairity...')
    if len(embedding) <1 :
        return None
    embedding = torch.cat(embedding, dim=0).cuda()
    # embedding = torch.cat(embedding, dim=0)
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    print(n, len(embedding))
    assert len(embedding) == n
    embedding = F.normalize(embedding, dim=0)
    pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
    gt = id_labels.expand(n,n).eq(id_labels.expand(n,n).t()).numpy()
    
    up_triangle = np.where(np.triu(pdist)- np.eye(n)*pdist !=0)
    pdist = pdist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far,tar,threshold = metrics.roc_curve(gt, pdist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    for f,fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
    return tar_at_far


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--weights_path', type=str, default='../weights/trained/2/', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--print-interval', type=int, default=10, help='size of each image dimension')
    parser.add_argument('--test-emb', action='store_true', help='test embedding')
    parser.add_argument('--test-trainset', action='store_true', help='test trainset')
    
    opt = parser.parse_args()
    print(opt, end='\n\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    weights_path = opt.weights_path
    with open(os.path.join(weights_path, 'model.yaml'), 'r') as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)
    weights = os.path.join(weights_path, 'latest.pt')
    img_size = (cfg['width'], cfg['height'])
    backbone_name = cfg['backbone_name']
    # Initialize model
    backbone = resnet_fpn_backbone(backbone_name, True)
    backbone.out_channels = 256
    # model = FRCNN_FPN(num_classes=2)

    model = Jde_RCNN(backbone, num_ID=cfg['num_ID'], min_size=img_size[1], max_size=img_size[0])
    # model = torch.nn.DataParallel(model)
    checkpoint = torch.load(weights, map_location='cpu')['model']
    # Load weights to resume from
    print(model.load_state_dict(checkpoint, strict=False))
    # model.load_state_dict(checkpoint)
    model.cuda().eval()
    # model.eval()
    # Get dataloader
    root = '/data/dgw'
    # root = '/home/hunter/Document/torch'
    if not opt.test_trainset:
        # paths = {'CP_val':'./data/detect/cp_val.txt',
        #         'M16':'./data/detect/MOT16_val.txt',
        #         'PRW':'./data/detect/PRW_val.txt',
        #         'CT':'./data/detect/CT_val.txt'}
        paths = {'M16':'./data/detect/MOT16_val.txt'}
    else:
        #paths = {'CP':'./data/detect/cp_train.txt',
        #        'M16':'./data/detect/MOT16_train.txt',
        #        'PRW':'./data/detect/PRW_train.txt',
        #        'CT':'./data/detect/CT_train.txt'}
        paths = {'M16':'./data/detect/MOT16_train.txt'}
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(root=root, paths=paths, img_size=img_size, augment=False, transforms=transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    with torch.no_grad():
        if opt.test_emb:
            res = test_emb(
                model,
                dataloader,
                opt.iou_thres,
                opt.conf_thres,
                opt.nms_thres,
                opt.print_interval,
            )
        else:
            mAP = test(
                model,
                dataloader,
                opt.iou_thres,
                opt.conf_thres,
                opt.nms_thres,
                opt.print_interval,
            )

