import os
import time
import torch
import numpy as np
import gc
import codecs

import scipy.io as sio
import json
import argparse

from PIL import Image

from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt
from synthgen import RendererV3 

from tqdm import tqdm

from pycocotools import mask
from skimage import measure
from itertools import groupby

PRECOMP_PATH = './datasets/detection/'
DICT_PATH = './datasets/detection/blurred_dict.json'

CHUNK = 1350
NUM_STAGES = 20
NUM_WORKERS = 10

parser = argparse.ArgumentParser()

parser.add_argument("part", type=int)
parser.add_argument("worker", type=int)


def binary_mask_to_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask)
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def open_file(string):
    with open(string, 'r') as f:
        ans = f.readlines()
        for i in range(len(ans)):
            ans[i] = ans[i].strip()
    return ans

def open_dict(string):
    with open(string, 'r') as f:
        curr_dict = json.load(f)
    return curr_dict


def main(curr_part, curr_worker):
    PART = curr_part
    WORKER = curr_worker
    RV3 = RendererV3('./data',max_time=10)
    blur_dict = open_dict(DICT_PATH)
    for stage in range(NUM_STAGES):
        part_start = WORKER * CHUNK * NUM_STAGES + CHUNK * stage
        part_end = WORKER * CHUNK * NUM_STAGES + CHUNK * (stage + 1)
        names = [item[item.find('l'):-4] for item in os.listdir(os.path.join(PRECOMP_PATH,
                                                                             f'seg_in/seg_{PART}'))][part_start:part_end]
        img_paths = [blur_dict[name] for name in names]
        
        test_seg = []
        for i in tqdm(range(len(names)), desc='input segs'):
            test_seg.append(sio.loadmat(f'{PRECOMP_PATH}seg_in/seg_{PART}/{names[i]}.mat'))
        
        test_dep = []
        for i in tqdm(range(len(names)), desc='input deps'):
            test_dep.append(sio.loadmat(f'{PRECOMP_PATH}dep_in/dep_{PART}/{names[i]}.mat'))
            
        test_imgs = []
        for i in tqdm(range(len(img_paths)), desc='input_imgs'):
            name = img_paths[i]
            img = np.array(Image.open(name))
            if len(img.shape) == 2:
                img = np.tile(img, (3, 1, 1))
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            assert len(img.shape) == 3
            assert img.shape[2] == 3
            test_imgs.append(img)
            
        test_results = []
        imnames = []
        test_data = list(zip(test_imgs, test_dep, test_seg, names))
        gc.collect()
        
        for i in tqdm(range(len(test_data)), desc='generating'):
            img, dep_dict, seg_dict, imname = test_data[i]
            dep = dep_dict['dep']
            seg, area, label = seg_dict['mask'], seg_dict['areas'], seg_dict['labels']
            img = Image.fromarray(img)

            sz = dep.shape[:2][::-1]

            img = np.array(img.resize(sz,Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

            res = RV3.render_text(img, dep, seg, area, label, ninstance=1, viz=False)
            
            if (res is not None) and (len(res) > 0):# and (check_difference(img, res[0]['img'])):
                test_results.append(res)
                imnames.append(imname)
        
        test_dep = []
        test_seg = []
        test_imgs = []
        gc.collect()
        
        data_dicts = []
        for i in tqdm(range(len(test_results)), desc='saving'):
            elem = test_results[i]
            try:
                kek = Image.fromarray(elem[0]['img'])
                new_seg = binary_mask_to_rle(elem[0]['seg'])
                
                kek.save(.save(f'{PRECOMP_PATH}img_out/img_{PART}/{imnames[i]}.jpg'))

                new_dict = {}
                new_dict['seg'] = new_seg
                new_dict['text'] = elem[0]['txt']
                new_dict['wordBB'] = elem[0]['wordBB'].tolist()
                new_dict['charBB'] = elem[0]['charBB'].tolist()
                data_dicts.append(new_dict)
            except:
                print(f'error in {i}th image')
                
        json_data = {k:v for k, v in zip(imnames, data_dicts)}
        file_path = f'{PRECOMP_PATH}jsons/json_{PART}/json_{WORKER}_{stage}.json'
        print(stage)
        json.dump(json_data, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.part, args.worker)