#%%
import torch
from utils.datasets import LoadImagesAndLabels, collate_fn, JointDataset
from torchvision.transforms import transforms as T
#%%
root = '.'
paths = {'CT':'./data/CalTech.txt', 'CUHK':'./data/CUHK.txt', 'ETH':'./data/ETH.txt', 'M16':'./data/MOT16_train.txt', 'PRW':'./data/PRW.txt', 'CP':'./data/cp_train.txt'}
transforms = T.Compose([T.ToTensor()])
trainset = JointDataset(root=root, paths=paths, augment=True, transforms=transforms)

# %%
