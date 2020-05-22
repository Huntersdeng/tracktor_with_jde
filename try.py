import os
import numpy as np

path = '../dataset/2DMOT15/test/'
seq_paths = os.listdir(path)
for seq_path in seq_paths:
    with open(os.path.join('./data/track/test/',seq_path+'.txt'), 'w+') as file:
        current_path = os.path.join(path,seq_path+'/images')
        for img_path in os.listdir(current_path):
            file.writelines(os.path.join(current_path.lstrip('../'),img_path)+'\n')