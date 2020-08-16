import os
from collections import defaultdict
import logging
import time

import cv2
import numpy as np

pipe_logger = logging.getLogger("pipeline")

class NoInstancesFoundError(Exception):
    pass

class NoSubjectFoundError(Exception):
    pass

class Pipeline:
    def __init__(self, masking_stage, trimap_stage, refinement_stage, 
                feedback_thresh=0.01):
        self.masking_stage = masking_stage
        self.trimap_stage = trimap_stage
        self.refinement_stage = refinement_stage
        self.feedback_thresh = feedback_thresh
        self.results = defaultdict(list)

        pipe_logger.info("Alpha feedback iteration threshold: {}".format(feedback_thresh))


    def __call__(self, img, annotated_img=None):
        """
        Performs end-to-end object extraction process process
        Args:
            img (numpy array): The original image
            annotated_img (numpy array):
                Annnotations made for specific object selection. Same shape as `img`
                * Foreground pixels represented by 1
                * Background pixels represented by 0
                * Unannotated pixels represented by -1
        Returns:
            dict(list):
                Intermediate and final image processing results
                * Initial instance segmentation
                * Foreground/Background masks used for initial trimap
                * Subsequent trimaps (from alpha feedback loop)
                * Alpha/Foreground prediction(s)
                * Matte(s)
        """
        try:
            heatmap, bounding_box = self.to_masking_stage(img, annotated_img)
        except NoSubjectFoundError:
            return self.results
        except NoInstancesFoundError:
            return None
        
        trimap = self.to_trimap_stage(heatmap, img, bounding_box, annotated_img)

        alpha = self.to_refinement_stage(trimap, img)
        
        self.to_refinement_loop(trimap, alpha, img)

        return self.results


    def to_masking_stage(self, img, annotated_img=None):
        """
        Performs instance detection and identification of the subject's 
        probability mask (heatmap)
        Args:
            img (numpy array): The original image
            annotated_img (numpy array):
                Annnotations made for specific object selection. Same shape as `img`
                * Foreground pixels represented by 1
                * Background pixels represented by 0
                * Unannotated pixels represented by -1
        Returns:
            (numpy array): Subject's probability mask 
            (list): Subject's bounding box offset such that
                    height = bounding_box[3] - bounding_box[1]
                    width = bounding_box[2] - bounding_box[0])
        """
        instances = self.masking_stage.get_instance_preds(img)
        self.results['instances'] = self.masking_stage.get_instances_vis(instances, img)
        
        heatmap, bounding_box = self.masking_stage.get_subject(instances, img, annotated_img)
        self.results['heatmap'] = heatmap*255

        return heatmap, bounding_box


    def to_trimap_stage(self, heatmap, img, bounding_box, annotated_img=None):
        """
        Generates the trimap given a subject of interest's probability mask and 
        bounding box offset.
        Args:
            heatmap (numpy array): The suject's instance probabilty mask.
            img (numpy array): The original image.
            bounding_box (list):
                Subject's bounding box offset such that
                height = bounding_box[3] - bounding_box[1]
                width = bounding_box[2] - bounding_box[0])
            annotated_img (numpy array)
                Annnotations made for specific object selection. Same shape as `img`
                * Foreground pixels represented by 1
                * Background pixels represented by 0
                * Unannotated pixels represented by -1
        Returns:
            (numpy array): Trimap
                Definite background represented by 0
                Definite foreground represented by 1
                Unknown region represented by 0.5
        """
        trimap = self.trimap_stage.get_trimap(heatmap, img, bounding_box.astype(int), annotated_img)
        self.results['trimaps'].append(trimap*255)
        
        return trimap


    def to_refinement_stage(self, trimap, img):     
        """
        Performs alpha and foreground estimation given a trimap.
        Args:
            trimap (numpy array): 
                Definite background represented by 0
                Definite foreground represented by 1
                Unknown region represented by 0.5
            img (numpy array): The original image

        Returns:
            (numpy array): Alpha matte
                Floating point values between 0 and 1
        """  
        fg, alpha, matte = self.refinement_stage.process(trimap, img)
        self.results['foregrounds'].append(fg*255)
        self.results['alphas'].append(alpha*255)
        self.results['mattes'].append(matte*255)
        
        return alpha


    def to_refinement_loop(self, trimap, alpha, img):
        """
        Iteratively refines foreground and alpha estimation until the change
        in the unknown region of trimap falls below a threshold.
        Args:
            trimap (numpy array): 
                Definite background represented by 0
                Definite foreground represented by 1
                Unknown region represented by 0.5
            alpha (numpy array): 
                Floating point values between 0 and 1
            img (numpy array): The original image
        """  
        prev_def_area = np.sum(np.logical_or(trimap==1.0, trimap==0.0))

        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        
        new_def_area = np.sum(np.logical_or(trimap==1.0, trimap==0.0))
        def_area_delta = (new_def_area - prev_def_area)/prev_def_area

        if def_area_delta > self.feedback_thresh:
            self.results['trimaps'].append(trimap*255)
            alpha = self.to_refinement_stage(trimap, img)
            self.to_refinement_loop(trimap, alpha, img)





