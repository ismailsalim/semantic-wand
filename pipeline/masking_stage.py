# application libraries
from masking_network.predictor import Predictor
from masking_network.models import ModifiedRCNN

# external libraries
import torch
import torchvision
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.layers.mask_ops import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom

from detectron2.utils.visualizer import _create_text_labels, GenericMask, ColorMode

import copy 

class MaskingStage:
    def __init__(self, cfg, roi_score_threshold, mask_threshold):
        self.cfg = get_cfg() 
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_threshold
        self.cfg.MODEL.META_ARCHITECTURE = 'ModifiedRCNN'
        self.mask_threshold = mask_threshold

        self.results = None


    def get_all_instances(self, img):   
        # predictor = Predictor(self.cfg, self.mask_thresholds)
        predictor = Predictor(self.cfg, self.mask_threshold)
        preds = predictor(img)
        
        # only looking at first image in preds output for now
        # instances = preds[0]['instances'].to('cpu')
        instances = preds[0]['instances']
        if len(instances) == 0:
            raise NoInstancesDetectedError("No instances found during masking stage!")

        return instances


    def get_subject(self, instances, img, annotated_img=None):
        if annotated_img is not None:
            instance_masks = self.upscale_mask(instances, 0.5)
            idx = self.fg_intersection(instance_masks, annotated_img)
            if idx == None:
                raise NoSubjectFoundError("Can't find an object to extract!")
            
        else: # find instance with largest predicted box area in the image
            idx = instances.get('pred_boxes').area().argmax().item()
            
        subject = instances[idx]

        self.results = subject
        # self.results = subject.to('cpu')
        
        pred_box = subject.get('pred_boxes').tensor.cpu().numpy()[0]
        height = pred_box[3] - pred_box[1]
        width = pred_box[2] - pred_box[0]
        subject_area = (height, width) 

        # mask_ids = ['pred_mask_'+str(thresh) for thresh in self.mask_thresholds]
        # masks = [subject.get(mask_id).cpu().numpy().squeeze().astype(float)
        #         for mask_id in mask_ids]
        
        # return masks[0], masks[1], box_dim
        return subject, subject_area


    def fg_intersection(self, instances, annotated_img):
        # One instance mask with the largest intesersection with foreground annotated
        # pixels gets selected as the subject (fg pixels annotated as 1)
        masks = instances.cpu().numpy().astype(int) # boolean array (True -> part of mask)
        matched_pixels = [np.sum(np.logical_and(m==True, annotated_img==1)) for m in masks]
        
        if all(sum == 0 for sum in matched_pixels):
            return None # none of the instances have been annotated

        return matched_pixels.index(max(matched_pixels))
          


    # def visualise_instances(self, img, instances):
    #     all_masks = retry_if_cuda_oom(paste_masks_in_image)(
    #     instances.pred_masks[:, 0, :, :],  # N, 1, M, M
    #     instances.pred_boxes,
    #     instances.image_size,
    #     threshold=0.5,
    #     )
    #     instances.set("all_masks", all_masks)

    #     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)

    #     boxes = instances.pred_boxes.to('cpu')
    #     scores = instances.scores.to('cpu')
    #     classes = instances.pred_classes.to('cpu')
    #     labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

    #     masks = np.asarray(instances.get("all_masks").to('cpu'))
    #     masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        
    #     colors = [v._jitter([x/255 for x in v.metadata.thing_colors[c]]) for c in classes]
    #     alpha = 0.8

    #     v.overlay_instances(
    #         masks=masks,
    #         boxes=boxes,
    #         labels=labels,
    #         assigned_colors=colors,
    #         alpha=alpha,
    #     )

    #     return v.output.get_image()[:, :, ::-1]
    #     # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
    #     # v = v.draw_instance_predictions(instances.to('cpu'))


    #     # return v.get_image()[:, :, ::-1]


    # def visualise_mask(self, img, subject, region):
    #     all_masks = retry_if_cuda_oom(paste_masks_in_image)(
    #     instances.pred_masks[:, 0, :, :],  # N, 1, M, M
    #     instances.pred_boxes,
    #     instances.image_size,
    #     threshold=0.5,
    #     )

    #     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)

    #     boxes = subject.pred_boxes.to('cpu')
    #     scores = subject.scores.to('cpu')
    #     classes = subject.pred_classes.to('cpu')
    #     labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

    #     # if mask == 'unknown':
    #     #     mask_id = 'pred_mask_'+str(self.mask_thresholds[0])
    #     # elif mask == 'fg':
    #     #     mask_id = 'pred_mask_'+str(self.mask_thresholds[1])

    #     mask_id = 'pred_mask_'+str(0.2)
    #     self.results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
    #     self.results.pred_masks[0, :, :],  # N, 1, M, M
    #     self.results.pred_boxes,
    #     self.results.image_size,
    #     threshold=0.2,
    #     )


    #     masks = np.asarray(subject.get(region))
    #     # masks = np.asarray(self.results.get(''))
    #     masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        
    #     colors = [v._jitter([x/255 for x in v.metadata.thing_colors[c]]) for c in classes]
    #     alpha = 0.8

    #     v.overlay_instances(
    #         masks=masks,
    #         boxes=boxes,
    #         labels=labels,
    #         assigned_colors=colors,
    #         alpha=alpha,
    #     )

    #     return v.output.get_image()[:, :, ::-1]

    def visualise(self, preds, img, threshold):
        masks = self.upscale_mask(preds, threshold)

        # field = "mask_"+str(threshold)
        # preds.set(field, masks)

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)

        boxes = preds.pred_boxes.to('cpu')
        scores = preds.scores.to('cpu')
        classes = preds.pred_classes.to('cpu')
        labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

        masks = np.asarray(masks.to('cpu'))
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
        
        colors = [v._jitter([x/255 for x in v.metadata.thing_colors[c]]) for c in classes]
        alpha = 0.8

        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )

        return v.output.get_image()[:, :, ::-1]


    def upscale_mask(self, preds, threshold):
        return retry_if_cuda_oom(paste_masks_in_image)(preds.pred_masks[:, 0, :, :], 
                                                        preds.pred_boxes, 
                                                        preds.image_size, 
                                                        threshold=threshold)
        # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        # v = v.draw_instance_predictions(instances.to('cpu'))


    # return v.get_image()[:, :, ::-1]
class NoInstancesDetectedError(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoSubjectFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)

# def mask_visualisation(masking_preds, threshold):
#     all_masks = retry_if_cuda_oom(paste_masks_in_image)(
#     instances.pred_masks[:, 0, :, :],  # N, 1, M, M
#     instances.pred_boxes,
#     instances.image_size,
#     threshold=0.5,
#     )
#     instances.set("all_masks", all_masks)

#     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)

#     boxes = instances.pred_boxes.to('cpu')
#     scores = instances.scores.to('cpu')
#     classes = instances.pred_classes.to('cpu')
#     labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

#     masks = np.asarray(instances.get("all_masks").to('cpu'))
#     masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
    
#     colors = [v._jitter([x/255 for x in v.metadata.thing_colors[c]]) for c in classes]
#     alpha = 0.8

#     v.overlay_instances(
#         masks=masks,
#         boxes=boxes,
#         labels=labels,
#         assigned_colors=colors,
#         alpha=alpha,
#     )

#     return v.output.get_image()[:, :, ::-1]
#     # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
#     # v = v.draw_instance_predictions(instances.to('cpu'))


#     # return v.get_image()[:, :, ::-1]