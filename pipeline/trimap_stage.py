# system libraries
import logging
import time

# external libraries
import numpy as np
import cv2
import itertools

import detectron2
from detectron2.layers.mask_ops import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom

class TrimapStage: 
    def __init__(self, def_fg_thresholds, unknown_thresholds):
        self.def_fg_thresholds = sorted(def_fg_thresholds)
        self.unknown_thresholds = sorted(unknown_thresholds, reverse=True)
        # self.dilation_sf = dilation_sf 
        # self.k_size = k_size
        # self.k_shape = k_shape      

    def process_subject(self, subject, annotated_img=None):
        # if annotated_img is not None:  
        #     fg_mask = self.get_fg_mask(subject, thresholds['fg_thresh'], annotated_img)
        #     unknown_mask = self.get_unknown_mask(subject, thresholds['unknown_thresh'], annotated_img)
        # else:
        #     fg_mask = self.get_fg_mask(subject, min(thresholds['fg_thresh']))
        #     unknown_mask = self.get_fg_mask(subject, max(thresholds('unknown_thresh')))
        fg_mask, fg_thresh = self.get_fg_mask(subject, annotated_img)
        unknown_mask, unknown_thresh = self.get_unknown_mask(subject, annotated_img)

        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5

        if annotated_img is not None:
            trimap[annotated_img == 1] = 1.0
            trimap[annotated_img == 0] = 0.0

        return trimap, fg_thresh, unknown_thresh


    def get_fg_mask(self, subject, annotated_img=None):
        if annotated_img is not None:
            fg_masks = []
            for thresh in self.def_fg_thresholds:
                fg_masks.append(self.convert_to_binary_mask(subject, thresh))
            fg_mask, idx = self.find_optimal_mask(fg_masks, annotated_img, fg=True)
            threshold = self.def_fg_thresholds[idx]
        else:
            threshold = min(self.def_fg_thresholds)
            fg_mask = self.convert_to_binary_mask(subject, threshold) 

        return fg_mask, threshold


    def get_unknown_mask(self, subject, annotated_img=None):
        if annotated_img is not None:
            unknown_masks = []
            for thresh in self.unknown_thresholds:
                unknown_masks.append(self.convert_to_binary_mask(subject, thresh))
            unknown_mask, idx = self.find_optimal_mask(unknown_masks, annotated_img)
            threshold = self.unknown_thresholds[idx]
        else:
            threshold = max(self.unknown_thresholds)
            unknown_mask = self.convert_to_binary_mask(subject, threshold) 

        return unknown_mask, threshold


    def find_optimal_mask(self, masks, annotated_img, fg=False):
        if fg:
            return self.minimal_bg_intersection(masks, annotated_img)
        else:
            return self.most_fg_intersection(masks, annotated_img)


    def minimal_bg_intersection(self, masks, annotated_img):
        intersection = [np.sum(np.logical_and(mask==True, annotated_img==0.0)) for mask in masks]
        print("Minimal BG Intersections:", intersection)
        idx = intersection.index(min(intersection)) # find first minimum
        print("Minimal idx", idx)
        return masks[idx], idx 


    def most_fg_intersection(self, masks, annotated_img):
        intersection = [np.sum(np.logical_and(mask==True, annotated_img==1.0)) for mask in masks]
        print("Most FG Intersections:", intersection)
        idx = intersection.index(max(intersection)) # find first maximum
        print("Max idx,", idx)
        return masks[idx], idx 


    def convert_to_binary_mask(self, subject, thresh):
        binary_mask = retry_if_cuda_oom(paste_masks_in_image)(
                subject.pred_masks[:, 0, :, :],  # N, 1, M, M
                subject.pred_boxes,
                subject.image_size,
                thresh,
        )
        return binary_mask.cpu().numpy().squeeze()



    def process_alpha(self, alpha, trimap, level):
        start = time.time()

        # logging.debug("Dilation Scale Factor {}".format(self.dilation_sf))
        # # logging.debug("Kernel size: {}". format(self.k_size))
        # logging.debug("Kernel shape: {}".format(self.k_shape))        

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