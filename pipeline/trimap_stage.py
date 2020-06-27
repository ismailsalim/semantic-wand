# system libraries
import logging

# external libraries
import numpy as np
import cv2


class TrimapStage: 
    def __init__(self, k_scale_f, k_shape):
        self.k_scale_f = k_scale_f 
        self.k_shape = k_shape      

    def process_masks(self, fg_mask, unknown_mask):       
        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5
        return trimap

    def process_alpha(self, alpha, box_dim, iteration):
        logging.debug("Kernel scale factor: {}".format(self.k_scale_f))
        logging.debug("Kernel shape: {}".format(self.k_shape))
        logging.debug("Box dim: {}".format(box_dim))

        avg_dim = sum(box_dim)/2
        kernel = cv2.getStructuringElement(getattr(cv2, self.k_shape), (3, 3))                            

        dilated = cv2.dilate(alpha, kernel, iterations=int(0.01*avg_dim))
        eroded = cv2.erode(alpha, kernel, iterations=int(0.01*avg_dim))
        logging.debug("Erode/Dilate iterations: {}".format(int(0.01*avg_dim)))

        trimap = np.zeros(alpha.shape, dtype='float64')
        trimap.fill(0.5)
        trimap[eroded==1.0] = 1.0
        trimap[dilated==0.0] = 0.0

        return trimap