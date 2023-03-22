import os
import time
import torch
import numpy as np
import pandas as pd
import gc
import argparse

import scipy.io as sio
import json

from PIL import Image
from prep_scripts.floodFill import get_mask
from prep_scripts.ucm import UCM, filter_ucm

from multiprocessing.dummy import Pool

from tqdm import tqdm

ALL_IMPATHS = '/home/jovyan/markovivl/test_synthtext/SynthText/datasets/all_impaths.txt'

SAVE_PATH = '/home/jovyan/markovivl/test_synthtext/SynthText/datasets/detection'

WORKER = 0
CHUNK = 5000
STAGE = 0
NUM_STAGES = 1
NUM_WORKERS = 1

PART_START = (STAGE * NUM_WORKERS * CHUNK) + WORKER * CHUNK
PART_END = (STAGE * NUM_WORKERS * CHUNK) + (WORKER + 1) * CHUNK


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
    
# def normalize_depth(depth, m=2., std=0.64):
#     return (((depth - depth.mean()) / depth.std() ) * std) + m

def normalize_depth(depth):
    max_d = np.max(depth)
    min_d = np.min(depth)
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    depth[depth < 0] = 0
    depth[depth > 1] = 1
    return depth

def get_depth(img, inverse=True):
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    
    #img = cv2.normalize(cv2.resize(output, (400, 500)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #img = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
#     if inverse:
#         img = (255 - img)
    
#     new_depth = img.astype(np.float32)/255. + 1.031
    new_depth = (1 - normalize_depth(output)) * 60
    
    return new_depth

def clean_impath(name):
    num, folder = name.split('/')[-1], name.split('/')[-2]
    num = num.split('.')[0]
    f_e = folder.split('_')
    new_name = f'lhr_{f_e[3]}_{f_e[4]}_{num}'
    return new_name

def main():
    start = time.time()
    ucm_model = UCM()
    
    with open(ALL_IMPATHS, 'r') as f:
        img_names = f.readlines()[PART_START:PART_END]
    
    for i in range(len(img_names)):
        img_names[i] = img_names[i].strip()
    
    #load images
    imgs = []
    for i in tqdm(range(len(img_names))):
        name = img_names[i]
        img = np.array(Image.open(name))
        if len(img.shape) == 2:
            img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        imgs.append(img)
    
    names = [clean_impath(path) for path in img_names]
    
    #load depths
    depths = []
    for i in tqdm(range(len(imgs))):
        depths.append({'dep' : get_depth(np.array(imgs[i]))})
    
    #save depths
    for name, dep in zip(names, depths):
        sio.savemat(f'{SAVE_PATH}/dep_in/dep_4/{name}.mat', dep)
        
    depths = []
    gc.collect()
        
    #segs
    # # %%time
    ucms = [] 
    for i in tqdm(range(len(imgs))):
        ucms.append(ucm_model.get_hierarchy(imgs[i]))
    ucms = [filter_ucm(ucm, quantile=0.72) for ucm in ucms]
    gc.collect()
    
    pbar = tqdm(total=len(ucms))
    def process_seg(ucm):
        pbar.update(1)
        return get_mask(ucm, verbose=False)
    
    new_pool = Pool(processes=32)
    segs = new_pool.map(process_seg, ucms)
    new_pool.close()
    new_pool.join()
    new_pool.terminate()
    
    #save segs
    for name, seg in zip(names, segs):
        sio.savemat(f'{SAVE_PATH}/seg_in/seg_4/{name}.mat', seg)
    
    with open(f'{SAVE_PATH}/names_{WORKER}_{STAGE}.txt', 'w') as f:
        for name in names:
            f.write(f'{name}\n')
    end = time.time()
    print((end - start)/60)

            
if __name__ == "__main__":
    
    main()