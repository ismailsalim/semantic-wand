from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

import os
import time
import logging
from collections import defaultdict

import cv2
import numpy as np

class Pipeline:
    def __init__(self, max_img_dim = 1000,
                        mask_config = 'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        roi_score_threshold = 0.8,
                        mask_threshold = 0.5,
                        def_fg_thresholds = [0.97, 0.98, 0.99],
                        unknown_thresholds = [0.2, 0.15, 0.1],
                        # dilation_sf = 1000,
                        # kernel_size = 3,
                        # kernel_shape = 'MORPH_RECT',
                        matting_weights = './matting_network/FBA.pth',
                        iterations = 3):
        self.max_img_dim = max_img_dim
        self.masking_stage = MaskingStage(mask_config, roi_score_threshold, mask_threshold)
        self.trimap_stage = TrimapStage(def_fg_thresholds, unknown_thresholds)
        self.refinement_stage = RefinementStage(matting_weights)
        self.iterations = iterations
        self.results = defaultdict(list)


    def __call__(self, img, annotated_img=None):
        """
        Performs end-to-end matting process.
        Args:
            img (numpy array): The image to be processed.
                Same shape as `img`. 
                Annnotations made by the user to indicate object selection.
                Foreground pixels represented by 1.
                Background pixels represented by 0.
                Unannotated pixels represented by -1.
        Returns:
            dict(list):
                Intermediate and final image processing results.
                * Initial instance segmentation
                * Foreground/Background masks used for initial trimap
                * Subsequent trimaps (from alpha feedback loop)
                * Alpha/Foreground prediction(s)
                * Matte(s)
        """
        if annotated_img is not None:
            assert len(annotated_img.shape) == 2, "Annotation image must be grayscale!"
            assert img.shape[:2] == annotated_img.shape[:2], "Annotation image and input image must have same (h, w)!" 
                        
            annots = np.array([-1, 0, 1])
            assert not False in np.in1d(annotated_img, annots), "Anotated image must only contain [-1, 0, 1]"

        h, w = img.shape[:2]
        if h > self.max_img_dim or w > self.max_img_dim: 
            img = self.rescale(img) # can limit GPU usage
            if annotated_img is not None:
                annotated_img = self.rescale(annotated_img)

        subject, subject_area = self.to_masking_stage(img, annotated_img)

        trimap = self.to_trimap_stage(subject, img, annotated_img)

        alpha = self.to_refinement_stage(trimap, img)

        self.alpha_feedback(img, trimap, alpha, subject_area, 1)

        return self.results


    def to_masking_stage(self, img, annotated_img=None):
        start = time.time()

        instance_preds = self.masking_stage.get_all_instances(img)
        # self.results['mask_preds'] = instance_preds

        # instances_vis = self.masking_stage.visualise_instances(img, instance_preds)
        # self.results['instances'] = instance_preds
        self.results['instances'] = self.masking_stage.visualise(instance_preds, img, 0.5)
        
        subject, subject_area = self.masking_stage.get_subject(instance_preds, img, annotated_img)

        # self.results['subject'] = subject
        
        # unknown_mask_vis = self.masking_stage.visualise_mask(img, 'unknown')
        # fg_mask_vis = self.masking_stage.visualise_mask(img, 'fg')
        # self.results['unknown_mask'] = unknown_mask_vis
        # self.results['fg_mask'] = fg_mask_vis

        end = time.time()
        logging.debug("Masking stake took {} seconds".format(end-start))

        return subject, subject_area


    def to_trimap_stage(self, subject, img, annotated_img=None):
        start = time.time()

        trimap, fg_thresh, unknown_thresh = self.trimap_stage.process_subject(subject, annotated_img)
        # fg_mask_vis = self.masking_stage.visualise_mask(img, subject, "fg_mask")
        # unknown_mask_vis = self.masking_stage.visualise_mask(img, subject, "unknown_mask")
        
        self.results['fg_mask'] = self.masking_stage.visualise(subject, img, fg_thresh)
        self.results['unknown_mask'] = self.masking_stage.visualise(subject, img, unknown_thresh)
        self.results['trimaps'].append(trimap*255)

        end = time.time()
        logging.debug("Trimap stage took {} seconds".format(end-start))
        return trimap


    def to_refinement_stage(self, trimap, img):
        start = time.time()
        
        fg, alpha, matte = self.refinement_stage.process(trimap, img)
        self.results['foregrounds'].append(fg*255)
        self.results['alphas'].append(alpha*255)
        self.results['mattes'].append(matte*255)

        end = time.time()
        logging.debug("Matting stage took {} seconds".format(end-start))
        
        return alpha


    def alpha_feedback(self, img, trimap, alpha, box_dim, iteration):
        if iteration <= self.iterations:
            avg_dim = sum(box_dim)/2
            level = avg_dim/iteration
            logging.debug("Average box dim: {}".format(avg_dim))
            logging.debug("Process alpha level: {}".format(level))

            logging.debug('Beggining alpha iteration {}...'.format(iteration))
            trimap = self.trimap_stage.process_alpha(alpha, trimap, level)
            self.results['trimaps'].append(trimap*255)

            fg, alpha, matte = self.refinement_stage.process(trimap, img)
            self.results['foregrounds'].append(fg*255)
            self.results['alphas'].append(alpha*255)
            self.results['mattes'].append(matte*255)

            self.alpha_feedback(img, trimap, alpha, box_dim, iteration+1)


    def rescale(self, img):
        (h, w) = img.shape[:2]

        if h > w:
            r = self.max_img_dim/float(h)
            dim = (int(w*r), self.max_img_dim)
        else:
            r = self.max_img_dim/float(w)
            dim = (self.max_img_dim, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


