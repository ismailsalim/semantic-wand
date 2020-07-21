import os
from collections import defaultdict

import cv2
import numpy as np

class Pipeline:
    def __init__(self, masking_stage, trimap_stage, refinement_stage, 
                feedback_thresh=0.3, max_img_dim=1000):
        self.masking_stage = masking_stage
        self.trimap_stage = trimap_stage
        self.refinement_stage = refinement_stage

        self.max_img_dim = max_img_dim
        self.feedback_thresh = feedback_thresh
        self.results = defaultdict(list)


    def __call__(self, img, annotated_img=None):
        """
        Performs end-to-end matting process.
        Args:
            img (numpy array): The image to be processed.
            annotated_img (numpy array):
                Annnotations made for specific object selection. Same shape as `img`. 
                * Foreground pixels represented by 1.
                * Background pixels represented by 0.
                * Unannotated pixels represented by -1.
        Returns:
            dict(list):
                Intermediate and final image processing results.
                * Initial instance segmentation
                * Foreground/Background masks used for initial trimap
                * Subsequent trimaps (from alpha feedback loop)
                * Alpha/Foreground prediction(s)
                * Matte(s)
        """
        # (refactor) move out of demo logic
        if annotated_img is not None:
            assert len(annotated_img.shape) == 2, "Annotation image must be grayscale!"
            assert img.shape[:2] == annotated_img.shape[:2], "Annotation image and input image must have same (h, w)!" 
                        
            annots = np.array([-1, 0, 1])
            assert not False in np.in1d(annotated_img, annots), "Anotated image must only contain [-1, 0, 1]"

        # (refactor) move of of demo logic
        h, w = img.shape[:2]
        if h > self.max_img_dim or w > self.max_img_dim: 
            img = self._rescale_img(img) 
            if annotated_img is not None:
                annotated_img = self._rescale_img(annotated_img)

        subject, bounding_box = self.to_masking_stage(img, annotated_img)

        trimap = self.to_trimap_stage(subject, img, bounding_box, annotated_img)

        alpha = self.to_refinement_stage(trimap, img)
        
        self.to_refinement_loop(trimap, alpha, img)

        return self.results


    def to_masking_stage(self, img, annotated_img=None):
        instance_preds = self.masking_stage.get_all_instances(img)

        # (refactor) move this out of pipe
        self.results['instances'] = self.masking_stage.visualise(instance_preds, img, 0.5)
        
        subject, bounding_box = self.masking_stage.get_subject(instance_preds, img, annotated_img)
        return subject, bounding_box


    def to_trimap_stage(self, subject, img, bounding_box, annotated_img=None):
        heatmap, trimap, fg_mask, unknown_mask = self.trimap_stage.process_subject(subject, 
                                                                                img,                 
                                                                                bounding_box.astype(int),
                                                                                annotated_img)
        
        # (refactor) move this out of pipe
        self.results['heatmap'] = heatmap*255
        self.results['fg_mask'] = fg_mask*255
        self.results['unknown_mask'] = unknown_mask*255
        self.results['trimaps'].append(trimap*255)
        
        return trimap


    def to_refinement_stage(self, trimap, img):       
        fg, alpha, matte = self.refinement_stage.process(trimap, img)
        self.results['foregrounds'].append(fg*255)
        self.results['alphas'].append(alpha*255)
        self.results['mattes'].append(matte*255)
        
        return alpha


    def to_refinement_loop(self, trimap, alpha, img):
        prev_def_area = np.sum(np.logical_or(trimap==1.0, trimap==0.0))

        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        
        new_def_area = np.sum(np.logical_or(trimap==1.0, trimap==0.0))
        def_area_delta = (new_def_area - prev_def_area)/prev_def_area

        if def_area_delta > self.feedback_thresh:
            self.results['trimaps'].append(trimap*255)
            alpha = self.to_refinement_stage(trimap, img)
            self.to_refinement_loop(trimap, alpha, img)


    def _rescale_img(self, img):
        (h, w) = img.shape[:2]

        if h > w:
            r = self.max_img_dim/float(h)
            dim = (int(w*r), self.max_img_dim)
        else:
            r = self.max_img_dim/float(w)
            dim = (self.max_img_dim, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


