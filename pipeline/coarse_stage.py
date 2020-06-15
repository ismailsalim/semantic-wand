# external libraries
import torch
import torchvision
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class CoarseStage:
    def __init__(self, cfg, thresh):
        self.cfg = get_cfg() 
        self.cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/'+cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/'+cfg)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh


    def pred(self, img):
        predictor = DefaultPredictor(self.cfg)
        preds = predictor(img)

        if len(preds['instances']) == 0:
            raise InstanceError
        
        return preds 


    def get_instances(self, img, preds):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(preds['instances'].to('cpu'))
        return v.get_image()[:, :, ::-1]


    def get_subj_mask(self, preds):
        main_instance = self.find_subj(preds['instances']) 
        mask = main_instance.get('pred_masks').cpu().numpy().squeeze().astype(float)
        size = main_instance.get('pred_boxes').area().cpu().item()
        return mask, size
    

    def find_subj(self, instances):
        if len(instances) == 0:
            return np.zeros(instances.image_size)
        
        max_area = max(instances.get('pred_boxes').area())
        main_subj = None
        for i in range(len(instances)):
            if instances[i].get('pred_boxes').area()==max_area:
                main_subj = instances[i]
                break    

        return main_subj


class InstanceError(Exception):
    """Thrown when no instances can be found by Mask R-CNN"""
    def __init__(self):
        msg = 'No instances are found during coarse stage!'
        super(InstanceError, self).__init__(msg)