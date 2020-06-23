import numpy as np
import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

@META_ARCH_REGISTRY.register()
class ModifiedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mask_thresholds):
        """                
        For running inference on given input images.
        Args:
            batched_inputs (list[dict]):
                Each item in the list is a dict that contains
                * image: Tensor, image in (C, H, W) format.
                * height (int), width (int): the output resolution of the model used 
                  for post-proessing
        Returns:
            list[dict]:
                Each dict is the modified Instances output for one input image
                including the definite foreground and unknown region masks.
        """
        images = self.preprocess_image(batched_inputs)
        
        features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, None)

        results, _ = self.roi_heads(images, features, proposals, None)

        return self.postprocess(results, batched_inputs, images.image_sizes, mask_thresholds)


    def preprocess_image(self, batched_inputs):
        """
        Normalizes, pads and batches the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def postprocess(self, instances, batched_inputs, image_sizes, mask_thresholds):
        """
        Rescales the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = self.detector_postprocess(results_per_image, height, width, mask_thresholds)
            processed_results.append({"instances": r})
        
        return processed_results


    def detector_postprocess(self, results, output_height, output_width, mask_thresholds):
        """
        Resizes the raw outputs of an R-CNN detector to produce outputs according to the 
        desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
            output_height, output_width: the desired output resolution.

        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        # Converts integer tensors to float temporaries to ensure true division is performed 
        # when computing scale_x and scale_y.
        if isinstance(output_width, torch.Tensor):
            output_width_tmp = output_width.float()
        else:
            output_width_tmp = output_width

        if isinstance(output_height, torch.Tensor):
            output_height_tmp = output_height.float()
        else:
            output_height_tmp = output_height

        scale_x, scale_y = (
            output_width_tmp / results.image_size[1],
            output_height_tmp / results.image_size[0],
        )
        results = Instances((output_height, output_width), **results.get_fields())

        # if results.has("pred_boxes"):
        #     output_boxes = results.pred_boxes
        # elif results.has("proposal_boxes"):
        #     output_boxes = results.proposal_boxes

        output_boxes = results.pred_boxes
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        # if results.has("pred_masks"):
        #     for thresh in mask_thresholds:
        #         field = "mask" + str(thresh)
        #         instances_masks = retry_if_cuda_oom(paste_masks_in_image)(
        #         results.pred_masks[:, 0, :, :],  # N, 1, M, M
        #         results.pred_boxes,
        #         results.image_size,
        #         thresh,
        #         )
        #         results.set(field, instances_masks)

        for thresh in mask_thresholds:
            field = "pred_mask_" + str(thresh)
            instances_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            thresh,
            )
            results.set(field, instances_masks)

        return results