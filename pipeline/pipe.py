from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

import os
import time
import logging
from collections import defaultdict

import cv2

class Pipeline:
    def __init__(self, max_img_dim = 1000,
                        mask_config = 'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        mask_thresh = 0.8,
                        trimap_thresholds = [0.2, 0.9],
                        kernel_scale_factor = 1000,
                        kernel_shape = 'MORPH_RECT',
                        matting_weights = './matting_network/FBA.pth',
                        iterations = 0):
        self.max_img_dim = max_img_dim
        self.masking_stage = MaskingStage(mask_config, mask_thresh, trimap_thresholds)
        self.trimap_stage = TrimapStage(kernel_scale_factor, kernel_shape)
        self.refinement_stage = RefinementStage(matting_weights)
        self.iterations = iterations
        self.results = defaultdict(list)


    def __call__(self, img):
        h, w = img.shape[:2]
        if h > self.max_img_dim or w > self.max_img_dim:
            img = self.rescale(img) 

        instance_preds = self.masking_stage.pred(img)
        instances_vis = self.masking_stage.visualise_instances(img, instance_preds)
        self.results['instances'] = instances_vis

        unknown_mask, fg_mask, box_dim = self.masking_stage.get_subject_masks(instance_preds)
        unknown_mask_vis = self.masking_stage.visualise_mask(img, 'unknown')
        fg_mask_vis = self.masking_stage.visualise_mask(img, 'fg')
        self.results['unknown_mask'] = unknown_mask_vis
        self.results['fg_mask'] = fg_mask_vis
 
        trimap = self.trimap_stage.process_masks(fg_mask, unknown_mask)
        self.results['trimaps'].append(trimap*255)

        fg, alpha, matte = self.refinement_stage.process(trimap, img)
        self.results['foregrounds'].append(fg*255)
        self.results['alphas'].append(alpha*255)
        self.results['mattes'].append(matte*255)

        self.alpha_feedback(img, alpha, box_dim, 0)

        return self.results


    def alpha_feedback(self, img, alpha, box_dim, iteration):
        if iteration < self.iterations: # loop until reaches user specification
            trimap = self.trimap_stage.process_alpha(alpha, box_dim, iteration)
            self.results['trimaps'].append(trimap*255)

            fg, alpha, matte = self.refinement_stage.process(trimap, img)
            self.results['foregrounds'].append(fg*255)
            self.results['alphas'].append(alpha*255)
            self.results['mattes'].append(matte*255)

            self.alpha_feedback(img, alpha, box_dim, iteration+1)


    def rescale(self, img):
        (h, w) = img.shape[:2]

        if h > w:
            r = self.max_img_dim/float(h)
            dim = (int(w*r), self.max_img_dim)
        else:
            r = self.max_img_dim/float(w)
            dim = (self.max_img_dim, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
