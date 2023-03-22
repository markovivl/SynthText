import random
import os
import numpy as np
from PIL import Image
import cv2


from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)


class RemoveText:
    
    def __init__(self, thres=5):
        self.refine_net = load_refinenet_model(cuda=True)
        self.craft_net = load_craftnet_model(cuda=True)
        self.thres = thres
        
    def get_text_heatmap(self, img):
        prediction_result = get_prediction(
            image=img,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=True,
            long_size=1280
        )
        
        heatmap = prediction_result['heatmaps']['text_score_heatmap'][:, :, 0]
        heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, np.ones((71, 71), np.uint8))
        heatmap = cv2.dilate(heatmap, np.ones((7, 7), np.uint8), iterations=2)
        
        heatmap = Image.fromarray(np.uint8(heatmap * 255))
        return np.array(heatmap.resize((img.shape[1], img.shape[0])))
        # return np.array(heatmap)
    
    def remove(self, img, heatmap):
        heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
        return np.clip(img - ((heatmap < self.thres) * 255), 0, 255)
    
    def blur(self, img, heatmap):
        heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
        blur = cv2.blur(img, (45, 45), 0)
        out = img.copy()
        # heatmap = np.array(Image.fromarray(heatmap).resize((out.shape[1], out.shape[0])))
        out[heatmap < self.thres] = blur[heatmap < self.thres]
        
        return out
