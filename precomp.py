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



ALL_IMPATHS = './datasets/all_impaths.txt'

PART_START = 0
PART_END = 10

SAVE_PATH = './datasets'


def normalize_depth(depth):
    max_d = np.max(depth)
    min_d = np.min(depth)
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    depth[depth < 0] = 0
    depth[depth > 1] = 1
    return depth

def get_depth(img):
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

    new_depth = (1 - normalize_depth(output)) * 60
    
    return new_depth

def clean_impath(name):
    num, folder = name.split('/')[-1], name.split('/')[-2]
    num = num.split('.')[0]
    f_e = num.split('_')
    new_name = f'ct_{f_e[-1]}'
    return new_name

def main():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
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
        sio.savemat(f'{SAVE_PATH}/dep_in/{name}.mat', dep)
        
    depths = []
    gc.collect()
        
    #segs
    # # %%time
    ucms = [] 
    for i in tqdm(range(len(imgs))):
        ucms.append(ucm_model.get_hierarchy(imgs[i]))
    ucms = [filter_ucm(ucm, quantile=0.72) for ucm in ucms]
    gc.collect()
    
    segs = [get_mask(ucm, verbose=False) for ucm in ucms]
    
    #save segs
    for name, seg in zip(names, segs):
        sio.savemat(f'{SAVE_PATH}/seg_in/{name}.mat', seg)
    end = time.time()
    print((end - start)/60)

            
if __name__ == "__main__":
    main()