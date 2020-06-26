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


    def pred(self, img):   
        predictor = Predictor(self.cfg, self.mask_thresholds)
        preds = predictor(img)
        
        # only looking at first image in preds output for now
        instances = preds[0]['instances'].to('cpu')
        if len(instances) == 0:
            raise InstanceError

        return instances


    def visualise_instances(self, img, instances):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        v = v.draw_instance_predictions(instances)
        return v.get_image()[:, :, ::-1]


    def get_subject_masks(self, instances):
        max_area = max(instances.get('pred_boxes').area())
        main_subject = None
        for i in range(len(instances)):
            if instances[i].get('pred_boxes').area()==max_area:
                main_subject = instances[i]
                self.results = main_subject.to('cpu')
                break    
        
        size = main_subject.get('pred_boxes').area().cpu().item()

        mask_ids = ['pred_mask_'+str(thresh) for thresh in self.mask_thresholds]
        masks = [main_subject.get(mask_id).cpu().numpy().squeeze().astype(float)
                for mask_id in mask_ids]
        
        return masks[0], masks[1], size


    def visualise_mask(self, img, mask_threshold):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        
        boxes = self.results.pred_boxes
        scores = self.results.scores
        classes = self.results.pred_classes
        labels = _create_text_labels(classes, scores, v.metadata.get("thing_classes", None))

        mask_id = "pred_mask_"+str(mask_threshold)
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


class InstanceError(Exception):
    """Thrown when no instances can be found by Mask R-CNN"""
    def __init__(self):
        msg = 'No instances are found during coarse stage!'
        super(InstanceError, self).__init__(msg)

