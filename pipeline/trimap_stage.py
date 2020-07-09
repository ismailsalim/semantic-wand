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
        self.def_fg_thresholds = def_fg_thresholds
        self.unknown_thresholds = unknown_thresholds


    def process_subject(self, subject, annotated_img=None):
        if annotated_img is not None:
            fg_mask, fg_thresh = self.get_mask(subject, self.def_fg_thresholds, min, annotated_img==0)
            unknown_mask, unknown_thresh = self.get_mask(subject, self.unknown_thresholds, max, annotated_img==1)
        else:
            fg_mask, fg_thresh = self.get_mask(subject, self.def_fg_thresholds, min)
            unknown_mask, unknown_thresh = self.get_mask(subject, self.unknown_thresholds, max)

        # fg_mask, fg_thresh = self.get_mask(subject, self.def_fg_thresholds, min, annotated_img==0)
        # unknown_mask, unknown_thresh = self.get_mask(subject, self.unknown_thresholds, max, annotated_img==1)

        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5

        if annotated_img is not None:
            trimap[annotated_img == 1] = 1.0
            trimap[annotated_img == 0] = 0.0

        return trimap, fg_thresh, unknown_thresh


    def get_mask(self, subject, thresholds, preference, target=None):
        print("Thresholds", thresholds)
        print("Preference", preference)
        print("Target", target==None)
        if target is not None:
            masks = [self.generate_mask(subject, thresh) for thresh in thresholds]
            matching = [np.sum(np.logical_and(mask==True, target==True)) for mask in masks]
            print("matching", matching)
            optimal_idx = matching.index(preference(matching))
            print("optimal_idx", optimal_idx)
            return masks[optimal_idx], thresholds[optimal_idx] 
        else:
            optimal_mask = self.generate_mask(subject, preference(thresholds))
            return optimal_mask, preference(thresholds)

    # def get_mask(self, subject, thresholds, preference, annotated_img=None):
    #     if annotated_img is not None:
    #         print("threshold", thresholds)

    #         masks = []
    #         for thresh in thresholds:
    #             masks.append(self.generate_binary_mask(subject, thresh))
    #         mask, idx = self.find_optimal_mask(masks, annotated_img, preference)
    #         threshold = thresholds[idx]
    #     else: # no additional user input to find better mask
    #         threshold = preference(thresholds) # just use general region's preference
    #         mask = self.generate_binary_mask(subject, threshold) 

    #     return mask, threshold


    def find_optimal_mask(self, masks, annotated_img, preference):
        annotations = [0, 1]
        target = preference(annotations)
        intersection = [np.sum(np.logical_and(mask==True, annotated_img==target)) for mask in masks]
        idx = intersection.index(preference(intersection)) # return first masks that satisfies preference
        print("preference", preference)
        print("intersection", intersection)
        print("idx", idx)
        return masks[idx], idx 


    def generate_mask(self, subject, thresh):
        binary_mask = retry_if_cuda_oom(paste_masks_in_image)(
            subject.pred_masks[:, 0, :, :],  # N, 1, M, M
            subject.pred_boxes,
            subject.image_size,
            thresh
        )
        return binary_mask.cpu().numpy().squeeze()


    def process_alpha(self, alpha, trimap, level):
        start = time.time()

        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0

        end = time.time()
        logging.debug('... took {} seconds'.format(end-start))
        return trimap