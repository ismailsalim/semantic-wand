# external libraries
import numpy as np
import cv2


class TrimapStage: 
    def __init__(self, ksize, dilation, erosion):
        if not ksize%2: # ksize can't be even
            raise ValueError('Kernel size must be odd!')
        self.ksize = ksize
        self.dilation = dilation
        self.erosion = erosion  
        

    def process(self, mask, size):
        base_its = int(size/10000)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(self.ksize,self.ksize))                            
        
        dilated = cv2.dilate(mask, kernel, iterations=base_its+self.dilation)
        eroded = cv2.erode(mask, kernel, iterations=base_its+self.erosion)
        
        trimap = np.zeros(mask.shape, dtype='float64')
        trimap.fill(0.5)
        trimap[eroded==1.0] = 1.0
        trimap[dilated==0.0] = 0.0
        return trimap