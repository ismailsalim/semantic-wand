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

    def process(self, mask, size, iteration):
        # To Do:
        # Heatmap
        # Iterations/Shape w/ Box Height

        # multiply kernel shape scale factor according to iteration number
        # formula is arbitrary (need to look into this)
        ksize = int(np.ceil(size/(self.k_scale_f*iteration))) // 2 * 2 + 1 
        
        logging.debug("Kernel scale factor: {}".format(self.k_scale_f))
        logging.debug("Kernel shape: {}".format(self.k_shape))
        logging.debug("Mask size: {}".format(size))
        logging.debug("Kernel size: {}".format(ksize))

        kernel = cv2.getStructuringElement(getattr(cv2, self.k_shape), (ksize,ksize))                            
        
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        trimap = np.zeros(mask.shape, dtype='float64')
        trimap.fill(0.5)
        trimap[eroded==1.0] = 1.0
        trimap[dilated==0.0] = 0.0
        
        return trimap

