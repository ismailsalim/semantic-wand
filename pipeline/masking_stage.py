from masking_network.predictor import Predictor
from masking_network.models import ModifiedRCNN

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


class NoInstancesDetectedError(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoSubjectFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MaskingStage:
    def __init__(self, cfg, roi_score_threshold, mask_threshold):
        # get default Detectron2 Mask R-CNN configuration
        self.cfg = get_cfg() 
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg)

        # custom configuration
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        self.cfg.MODEL.RPN.NMS_THRESH = 0.7
        self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
        
        self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_threshold       
        
        self.cfg.MODEL.META_ARCHITECTURE = 'ModifiedRCNN' # (refactor) pass in
        self.mask_threshold = mask_threshold
        self.predictor = Predictor(self.cfg, self.mask_threshold)


    def get_all_instances(self, img):
        preds = self.predictor(img)

        # (refactor) single/multiple image optimisation for eval
        instances = preds[0]['instances']
        if len(instances) == 0:
            raise NoInstancesDetectedError("No instances found during masking stage!")

        return instances


    def get_subject(self, instances, img, annotated_img=None):
        if annotated_img is not None:
            instance_masks = self._process_mask_probs(instances, 0.5)
            idx = self._most_annotated(instance_masks, annotated_img)
            if idx == None:
                raise NoSubjectFoundError("Can't find an object to extract!")

        # (refactor) allow this in demo (w/ select subject button)    
        else: # find instance with largest predicted box area in the image
            idx = instances.get('pred_boxes').area().argmax().item()
            
        subject = instances[idx]
        
        # (review) current used for alpha refinement loop
        pred_box = subject.get('pred_boxes').tensor.cpu().numpy()[0]
        return subject, pred_box


    def _most_annotated(self, instances, annotated_img):
        # Instance with mask that has largest intesersection with foreground annotated
        # pixels gets selected as the subject 
        masks = instances.cpu().numpy() # True implies image pixel is part of mask
        matching = [np.sum(np.logical_and(m==True, annotated_img==1)) for m in masks] 
        
        if all(sum == 0 for sum in matching):
            return None # none of the instances have been annotated

        return matching.index(max(matching))


    def visualise(self, preds, img, threshold):
        masks = self._process_mask_probs(preds, threshold)

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


    def _process_mask_probs(self, preds, threshold):
        return retry_if_cuda_oom(paste_masks_in_image)(
            preds.pred_masks[:, 0, :, :], 
            preds.pred_boxes, 
            preds.image_size, 
            threshold=threshold)