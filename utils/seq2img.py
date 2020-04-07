import os
import glob
import json
from scipy.io import loadmat
from collections import defaultdict

all_obj = 0
label_path = './myproject/dataset/CalTech/labels_with_ids/'
for dname in sorted(glob.glob('./myproject/dataset/CT/annotations/set*')):
    set_name = os.path.basename(dname)
    for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
        vbb = loadmat(anno_fn)
        objLists = vbb['A'][0][0][1][0]
        maxObj = int(vbb['A'][0][0][2][0][0])
        video_name = os.path.splitext(os.path.basename(anno_fn))[0]
        current_path = label_path+set_name+'/'+video_name
        n_obj = 0
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        for frame_id, obj in enumerate(objLists):
            if frame_id%10==0:
                frame_id = int(frame_id / 10)
                with open(current_path+'/'+str(frame_id)+'.txt', 'w+') as dst:
                    if len(obj) > 0:
                        for id, pos in zip(
                                obj['id'][0], obj['pos'][0]):
                            id = int(id[0][0]) - 1  # MATLAB is 1-origin
                            pos = pos[0].tolist()
                            pos[0] = pos[0] + pos[2]/2
                            pos[1] = pos[1] + pos[3]/2
                            dst.write(str(frame_id)+','+str(id)+','+','.join(str(x) for x in pos)+'\n')
                            # print(str(frame_id)+','+str(id)+','+','.join(str(x) for x in pos))
                            n_obj += 1

        print(dname, anno_fn, n_obj)
        all_obj += n_obj

print('Number of objects:', all_obj)

