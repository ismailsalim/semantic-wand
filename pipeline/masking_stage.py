from masking_network.predictor import Predictor
from masking_network.models import ModifiedRCNN

import logging

import torch
import torchvision
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import _create_text_labels, GenericMask, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.utils.memory import retry_if_cuda_oom

pipe_logger = logging.getLogger("pipeline")

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit
device = torch.device("cuda:0") 

class MaskingStage:
    def __init__(self, cfg="Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml", 
                        roi_score_threshold=0.05):
        
        self.cfg = get_cfg() 
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg)

        # R-CNN configuration specification
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        self.cfg.MODEL.RPN.NMS_THRESH = 0.7
        self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
        
        self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_threshold       
        
        self.cfg.MODEL.META_ARCHITECTURE = 'ModifiedRCNN' 
        
        self.predictor = Predictor(self.cfg)

        pipe_logger.info("ROI Score Threshold: {}".format(roi_score_threshold))


    def get_instance_preds(self, img):
        pipe_logger.info("Instance detection starting...")
        preds = self.predictor(img)

        instances = preds[0]['instances']
        if len(instances) == 0:
            raise ValueError("No instances found!")
        
        pipe_logger.info("Instances detected!")
        return instances


    def get_subject(self, instances, img, annotated_img=None):
        pipe_logger.info("Subject identification starting...")
        
        if annotated_img is not None:
            instance_masks = self._process_soft_masks(instances, 0.5)
            idx = self._get_most_annotated(instance_masks, annotated_img)                

        else: # instance with largest bounding box offset
            idx = instances.get('pred_boxes').area().argmax().item()
            
        subject = instances[idx]
        pipe_logger.info("Subject identified from annotations!")

        heatmap = self._process_soft_masks(subject, -1)

        bounding_box = subject.get('pred_boxes').tensor.cpu().numpy()[0]

        return heatmap, bounding_box


    def _get_most_annotated(self, instances, annotated_img):
        masks = instances.cpu().numpy() # boolean masks
        matching = [np.sum(np.logical_and(m==True, annotated_img==1)) for m in masks] 
        
        if all(sum == 0 for sum in matching):
            raise ValueError("Can't identify a subject from annotations!") 

        return matching.index(max(matching))


    def get_instances_vis(self, preds, img):
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)


        masks = self._process_soft_masks(preds, 0.5)
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        boxes = preds.pred_boxes.to('cpu')
        scores = preds.scores.to('cpu')
        classes = preds.pred_classes.to('cpu')
        labels = _create_text_labels(classes, scores, v.metadata.get('thing_classes', None))

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


    def _process_soft_masks(self, subject, threshold):
        # scale 28x28 soft mask output up to full image resolution
        binary_mask = retry_if_cuda_oom(self.paste_masks_in_image)(
            subject.pred_masks[:, 0, :, :],  
            subject.pred_boxes,
            subject.image_size,
            threshold = threshold
        )
        return binary_mask.cpu().numpy().squeeze()


    def paste_masks_in_image(self, masks, boxes, image_shape, threshold):        
        assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
        N = len(masks)
        if N == 0:
            return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
        if not isinstance(boxes, torch.Tensor):
            boxes = boxes.tensor
        device = boxes.device
        assert len(boxes) == N, boxes.shape

        img_h, img_w = image_shape

        if device.type == "cpu":
            num_chunks = N
        else:
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (
                num_chunks <= N
            ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        img_masks = torch.zeros(
            N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.float
        )
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)

            img_masks[(inds,) + spatial_inds] = masks_chunk
        return img_masks