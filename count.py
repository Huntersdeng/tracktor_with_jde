import os
import numpy as np 

paths_trainset = {
                    #'ETH':'./data/detect/train/ETH.txt',
                    'PRW':'./data/detect/train/PRW_train.txt', 
                    'CP':'./data/detect/train/cp_train.txt',
                    'CS':'./data/detect/train/CUHK_train.txt'}
paths_valset = {
                'CP':'./data/detect/val/cp_val.txt',
                'PRW':'./data/detect/val/PRW_val.txt',
                # 'CT':'./data/detect/val/CT_val.txt',
                'CS':'./data/detect/val/CUHK_val.txt'}

sum_frames = 0
sum_boxes = 0

for key in paths_trainset.keys():
    sum_files = []
    for path in [paths_trainset[key],paths_valset[key]]:
        with open(path, 'r') as file:
            img_files = file.readlines()
            img_files = [x.replace('\n', '') for x in img_files]
            img_files = list(filter(lambda x: len(x) > 0, img_files))
            sum_files += img_files
    num_frames = len(sum_files)
    num_boxes = 0
    sum_frames += num_frames
    print('nums of frames of '+key+': ', num_frames)
    for img_file in sum_files:
        label_file = img_file.replace('.jpg','.txt').replace('.png','.txt').replace('images','labels_with_ids')
        label = np.loadtxt(os.path.join('..',label_file), dtype=float, delimiter=',')
        num_boxes += label.shape[0]
    sum_boxes += num_boxes
    print('nums of boxes of '+key+': ', num_boxes)
print('sum of frames: ', sum_frames)
print('sum of boxes: ', sum_boxes)
