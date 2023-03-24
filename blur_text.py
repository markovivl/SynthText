import os
import time
import numpy as np
import pandas as pd

import scipy.io as sio
import json
from PIL import Image
import matplotlib.pyplot as plt
from remove_text import *
from tqdm import tqdm
import gc

TOTAL_IMAGE_NUM = 1134516

ALL_IMPATHS = './all_impaths.txt'

BLUR_PATH = './datasets/blurred_img/'


def get_new_impath(impath):
    '''
    rename your blurred images
    '''
    new_impath = BLUR_PATH + impath.split('/')[-1]
    return new_impath


def main():
    remove_text = RemoveText()
    with open(ALL_IMPATHS, 'r') as f:
        img_names = f.readlines()

    for i in range(len(img_names)):
        img_names[i] = img_names[i].strip()
    
    new_impaths = list(map(get_new_impath, img_names))
    
    loaded_images = []
    start_part = 0
    end_part = 10000
    while start_part < TOTAL_IMAGE_NUM:
        imgs = []
        gc.collect()
        for i in range(start_part, min(end_part, TOTAL_IMAGE_NUM)):
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

        for i in tqdm(range(len(imgs))):
            try:
                text_heatmap = remove_text.get_text_heatmap(imgs[i])
                imgs[i] = remove_text.blur(imgs[i], text_heatmap)
            except:
                print(f'blur error at {img_names[i]}')

        for i in range(len(imgs)):
            Image.fromarray(imgs[i]).save(new_impaths[start_part + i])

        loaded_images.extend(new_impaths[start_part:end_part])
        print(end_part)
        start_part += 10000
        end_part += 10000
    gc.collect()
    with open('loaded_images.txt', 'w') as f:
        for item in loaded_images:
            f.write(item + '\n')
        print('done')

if __name__ == "__main__":
    main()