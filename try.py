import os
import numpy as np

# path = '../dataset/2DMOT15/train/'
# seq_paths = os.listdir(path)
# for seq_path in seq_paths:
#     if not seq_path=='Venice-2':
#         continue
#     with open(os.path.join('./data/track/val/',seq_path+'.txt'), 'w+') as file:
#         current_path = os.path.join(path,seq_path+'/images')
#         for img_path in os.listdir(current_path):
#             file.writelines(os.path.join(current_path.lstrip('../'),img_path)+'\n')

# rootdir = '../dataset/2DMOT15/train/'
# for path in os.listdir(rootdir):
#     path = rootdir+path
#     gt = np.loadtxt(path+'/gt/gt.txt', dtype=float, delimiter=',', usecols=(0,1,2,3,4,5,6,7))
#     gt = gt[np.argsort(gt[:,0])]
#     gt = gt[gt[:,6]==1]
#     # print(gt.shape)
#     label = gt[:,1]
#     label_set = np.unique(label)
#     for i,id in enumerate(np.unique(label)):
#         label[label==id] = i
#     gt[:,1] = label
#     num_frames = int(np.max(gt[:,0]))
#     current_path = path+'/labels_with_ids/'
#     if not os.path.exists(current_path):
#         os.makedirs(current_path)
#     for i in range(num_frames):
#         src = gt[gt[:,0]==i+1][:,0:6]
#         src[:,2] = src[:,2] + src[:,4]/2
#         src[:,3] = src[:,3] + src[:,5]/2
#         np.savetxt(current_path+str(i+1).rjust(6,'0')+'.txt', src, fmt='%d', delimiter=',')

path = './data/detect/cp_train.txt'
root = './'
with open(path, 'r') as file:
    img_files = file.readlines()
    img_files = [x.replace('\n', '') for x in img_files]
    img_files = list(filter(lambda x: len(x) > 0, img_files))
label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                    for x in img_files]
num_frames = len(img_files)
from random import shuffle
shuffle(img_files)

img_train = img_files[:int(0.9*num_frames)]
img_val = img_files[int(0.9*num_frames):]
with open('./data/detect/train/cp_train.txt','w+') as file:
    for img_path in img_train:
         file.writelines(img_path+'\n')

with open('./data/detect/val/cp_val.txt','w+') as file:
    for img_path in img_val:
         file.writelines(img_path+'\n')
