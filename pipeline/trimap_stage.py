# system libraries
import logging
import time

# external libraries
import numpy as np
import cv2


class TrimapStage: 
    def __init__(self, dilation_sf, k_size, k_shape):
        self.dilation_sf = dilation_sf 
        self.k_size = k_size
        self.k_shape = k_shape      

    def process_masks(self, fg_mask, unknown_mask):       
        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5
        return trimap

    def process_alpha(self, alpha, trimap, level):
        start = time.time()

        logging.debug("Dilation Scale Factor {}".format(self.dilation_sf))
        # logging.debug("Kernel size: {}". format(self.k_size))
        logging.debug("Kernel shape: {}".format(self.k_shape))        

        # kernel = cv2.getStructuringElement(getattr(cv2, self.k_shape), 
        #                                     (self.k_size, self.k_size))                            

        # iterations = int(self.dilation_sf*level)
        # logging.debug("Dilation iterations: {}".format(iterations))       

        # dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        # eroded = cv2.erode(alpha, kernel, iterations=iterations)
        # logging.debug("Dilate iterations: {}".format(iterations))
        
        # trimap = np.zeros(alpha.shape, dtype='float64')
        # trimap[dilated==1.0] = 0.5
        # trimap[alpha==1.0] = 1.0
        # # trimap.fill(0.5)
        # # trimap[alpha==0]

        # trimap = np.zeros(alpha.shape, dtype='float64')
        # trimap[dilated > 0] = 0.5
        # trimap[np.logical_and(alpha>0, alpha<1)] = 0.5
        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0

        end = time.time()
        logging.debug('... took {} seconds'.format(end-start))
        return trimap