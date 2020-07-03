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

from detectron2.utils.visualizer import _create_text_labels, GenericMask, ColorMode


class MaskingStage:
    def __init__(self, cfg, thresh, mask_thresholds):
        self.cfg = get_cfg() 
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.cfg.MODEL.META_ARCHITECTURE = 'ModifiedRCNN'
        self.mask_thresholds = mask_thresholds

        self.results = None


    def get_all_instances(self, img):   
        predictor = Predictor(self.cfg, self.mask_thresholds)
        preds = predictor(img)
        
        # only looking at first image in preds output for now
        instances = preds[0]['instances'].to('cpu')
        if len(instances) == 0:
            raise ValueError("No instances found during masking stage!")

        return instances


    def get_subject(self, instances, annotated_img=None):
        if annotated_img is not None:
            instance_masks = instances.get('pred_masks')
            subject = self.fg_intersection(instances, annotated_img)
        else: # find instance with largest predicted box area in the image
            idx = instances.get('pred_boxes').area().argmax().item()
            subject = instances[idx]
        
        self.results = subject.to('cpu')
        
        pred_box = subject.get('pred_boxes').tensor.cpu().numpy()[0]
        height = pred_box[3] - pred_box[1]
        width = pred_box[2] - pred_box[0]
        box_dim = (height, width) 

        mask_ids = ['pred_mask_'+str(thresh) for thresh in self.mask_thresholds]
        masks = [subject.get(mask_id).cpu().numpy().squeeze().astype(float)
                for mask_id in mask_ids]
        
        return masks[0], masks[1], box_dim


    def fg_intersection(self, instances, annotated_img):
        # One instance mask with the largest intesersection with foreground annotated
        # pixels gets selected as the subject (fg pixels annotated as 1)
        masks = instances.get('pred_masks').cpu().numpy().astype(int) # boolean array (True -> part of mask)
        matched_pixels = [np.sum(m==annotated_img) for m in masks]
        most_matching_idx = matched_pixels.index(max(matched_pixels))
        print('masks:', masks)
        print('matched_pixels', matched_pixels)
        print('most_matching_idx', most_matching_idx)
        print('instances:', instances)
        print('instance[most_matching]', instances[most_matching_idx])
        return instances[most_matching_idx]


    def visualise_instances(self, img, instances):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        v = v.draw_instance_predictions(instances)
        return v.get_image()[:, :, ::-1]


    def visualise_mask(self, img, mask):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        
        boxes = self.results.pred_boxes
        scores = self.results.scores
        classes = self.results.pred_classes
        labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

        if mask == 'unknown':
            mask_id = 'pred_mask_'+str(self.mask_thresholds[0])
        elif mask == 'fg':
            mask_id = 'pred_mask_'+str(self.mask_thresholds[1])

        masks = np.asarray(self.results.get(mask_id))
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


