import cv2
import numpy as np
import higra as hg
from copy import deepcopy


import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())


TXT_PATH = 'deploy.prototxt'
MODEL_PATH = 'COB_PASCALContext_trainval.caffemodel'



def filter_ucm(ucm, quantile=None, hard_num = None):
    int_ucm = cv2.normalize(src=ucm, dst=None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    out_ucm = int_ucm.copy()

    if hard_num is not None:
        out_ucm[out_ucm < hard_num] = 0
    if quantile is not None:
        ucm_quantile = np.quantile(int_ucm[np.nonzero(int_ucm)], quantile)
        out_ucm[out_ucm <= ucm_quantile] = 0
    return out_ucm


class UCM:
    def __init__(self):
        self.num_clusters = 8
        self.mval = np.array([104.00698793, 116.66876762, 122.67891434, 104.00698793, 134.0313, 92.0228, 117.3808])
        self.max_res = 500
        self.model = cv2.dnn.readNetFromCaffe(TXT_PATH, MODEL_PATH)
#         self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#         self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        
    def caffe_forward(self, image):
        data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR).astype(np.float32)
        dim = data.shape
        flag = 0
        if ((dim[0] != self.max_res) or (dim[1] != self.max_res)):
            data = cv2.resize(data, [self.max_res, self.max_res])
            flag = 1
        assert data.shape[2] == 3
        for ch in range(data.shape[2]):
            data[:, :, ch] -= self.mval[ch]
        data = np.transpose(data, [1, 0, 2]) ## permute width and height
        data_blob = cv2.dnn.blobFromImage(data)
        self.model.setInput(data_blob)
        out = self.model.forwardAndRetrieve([
            'sigmoid-fuse_scale_2.0', 'sigmoid-fuse_scale_0.5', 'sigmoid-fuse_or8_1',
            'sigmoid-fuse_or8_2', 'sigmoid-fuse_or8_3', 'sigmoid-fuse_or8_4',
            'sigmoid-fuse_or8_5', 'sigmoid-fuse_or8_6', 'sigmoid-fuse_or8_7', 'sigmoid-fuse_or8_8'
                                 ])
        out = [elem[0][0].transpose(2, 1, 0) for elem in out]
        if flag:
            for i in range(len(out)):
                out[i] = cv2.resize(out[i], [dim[1], dim[0]], interpolation= cv2.INTER_NEAREST)
        else:
            out = np.array(out).squeeze()
                
        out_fine = out[0]
        out_coarse = out[1]
        out_or = np.array(out[2:]).transpose(1, 2, 0)
        return out_fine, out_coarse, out_or
    
    
    def trained_orientation(self, out_orientations):
        quant_angles = np.linspace(0, (self.num_clusters - 1) * np.pi / self.num_clusters, self.num_clusters)
        max_1 = np.max(out_orientations, axis=2)
        ind_1 = np.argmax(out_orientations, axis=2)
        
        m, n = ind_1.shape
        out_orientations[np.arange(m).reshape(-1, 1), np.arange(n), ind_1] = 0
        ind_1[max_1 < 0.01] = 0
        
        max_2 = np.max(out_orientations, axis=2)
        ind_2 = np.argmax(out_orientations, axis=2)

        ind_2[max_2 < 0.01] = 0
        
        O1 = quant_angles[ind_1]
        O2 = quant_angles[ind_2]
        return max_1, max_2, O1, O2
    
    
    def interpolate_angles(self, max_1, max_2, O1, O2):
        O1[(O2 == ((self.num_clusters - 1) * np.pi / self.num_clusters)) & (O1 == 0)] = np.pi
        O2[(O1 == ((self.num_clusters - 1) * np.pi / self.num_clusters)) & (O2 == 0)] = np.pi
        assert np.all(O1 >= 0) and np.all(O2 >= 0)
        #O3 = (max_1 * O1 + max_2 * O2)/(max_1 + max_2)
        O4 = (max_1 * O1 + max_2 * O2)/(max_1 + max_2 + 0.002)
        #O = np.where((np.abs((O1 - O2)) == (np.pi / self.num_clusters)), O3, O1)
        O = np.where((np.abs((O1 - O2)) == (np.pi / self.num_clusters)), O4, O1)
        return O
    
    
    def get_hierarchy(self, image):
        out_fine, out_coarse, out_or = self.caffe_forward(image)
        max_1, max_2, O1, O2 = self.trained_orientation(out_or)
        O = self.interpolate_angles(max_1, max_2, O1, O2)
        O[max_1 < 0.5] = -1
        O[O < 0] = np.pi * np.random.randint(1, np.sum(O < 0))
        
        image = image.astype(np.float64)/255 
        size = image.shape[:2]
        gradient_coarse = out_coarse
        gradient_fine = out_fine
        gradient_orientation = O
        
        graph = hg.get_4_adjacency_graph(size)
        edge_weights_fine = hg.weight_graph(graph, gradient_fine, hg.WeightFunction.mean)
        edge_weights_coarse = hg.weight_graph(graph, gradient_coarse, hg.WeightFunction.mean)

        # special handling for angles to wrap around the trigonometric cycle...
        edge_orientations_source = hg.weight_graph(graph, gradient_orientation, hg.WeightFunction.source) 
        edge_orientations_target = hg.weight_graph(graph, gradient_orientation, hg.WeightFunction.target) 
        edge_orientations = hg.mean_angle_mod_pi(edge_orientations_source, edge_orientations_target)
        
        
        combined_hierarchy, altitudes_combined = hg.multiscale_mean_pb_hierarchy(graph, edge_weights_fine, others_edge_weights=(edge_weights_coarse,), edge_orientations=edge_orientations)
        seg = hg.graph_4_adjacency_2_khalimsky(graph, hg.saliency(combined_hierarchy, altitudes_combined))
        
        
        return seg