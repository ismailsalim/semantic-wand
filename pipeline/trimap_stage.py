# system libraries
import logging

# external libraries
import numpy as np
import cv2


class TrimapStage: 
    def __init__(self, k_scale_f, k_shape, iterations):
        self.k_scale_f = k_scale_f 
        self.k_shape = k_shape      
        self.iterations = iterations

    def process(self, fg_mask, unknown_mask, size, iteration):
        # ksize = int(np.ceil(size/(self.k_scale_f*iteration))) // 2 * 2 + 1 
        
        # logging.debug("Kernel scale factor: {}".format(self.k_scale_f))
        # logging.debug("Kernel shape: {}".format(self.k_shape))
        # logging.debug("Mask size: {}".format(size))
        # logging.debug("Kernel size: {}".format(ksize))

        # kernel = cv2.getStructuringElement(getattr(cv2, self.k_shape), (ksize,ksize))                            
        
        # dilated = cv2.dilate(mask, kernel, iterations=1)
        # eroded = cv2.erode(mask, kernel, iterations=1)
        
        # trimap = np.zeros(mask.shape, dtype='float64')
        # trimap.fill(0.5)
        # trimap[eroded==1.0] = 1.0
        # trimap[dilated==0.0] = 0.0
        
        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5
        
        return trimap

