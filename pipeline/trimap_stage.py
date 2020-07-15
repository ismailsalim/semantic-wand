import numpy as np
import detectron2
import torch
from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.utils.memory import retry_if_cuda_oom

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


class TrimapStage: 
    def __init__(self, def_fg_thresholds, unknown_thresholds):
        self.def_fg_thresholds = def_fg_thresholds
        self.unknown_thresholds = unknown_thresholds


    def process_subject(self, subject, annotated_img=None):
        if annotated_img is not None:
            fg_mask, fg_thresh = self._get_mask(subject, self.def_fg_thresholds, min, annotated_img==0)
            unknown_mask, unknown_thresh = self._get_mask(subject, self.unknown_thresholds, max, annotated_img==1)
        else:
            fg_mask, fg_thresh = self._get_mask(subject, self.def_fg_thresholds, min)
            unknown_mask, unknown_thresh = self._get_mask(subject, self.unknown_thresholds, max)

        heatmap, _ = self._get_mask(subject)

        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5

        if annotated_img is not None:
            trimap[annotated_img == 1] = 1.0
            trimap[annotated_img == 0] = 0.0

        return heatmap, trimap, fg_mask, unknown_mask


    def process_alpha(self, alpha, trimap, level):
        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        return trimap


    def _get_mask(self, subject, thresholds=None, preference=None, target=None):
        if target is not None:
            masks = [self._generate_mask(subject, thresh) for thresh in thresholds]
            matching = [np.sum(np.logical_and(mask==True, target==True)) for mask in masks]
            optimal_idx = matching.index(preference(matching))
            return masks[optimal_idx], thresholds[optimal_idx] 
        elif preference is not None:
            optimal_mask = self._generate_mask(subject, preference(thresholds))
            return optimal_mask, preference(thresholds)
        else:
            heatmap = self._generate_mask(subject, -1)
            return heatmap, -1


    def _generate_mask(self, subject, thresh):
        binary_mask = retry_if_cuda_oom(self.paste_masks_in_image)(
            subject.pred_masks[:, 0, :, :],  # N, 1, M, M
            subject.pred_boxes,
            subject.image_size,
            thresh
        )
        return binary_mask.cpu().numpy().squeeze()


    def paste_masks_in_image(self, masks, boxes, image_shape, threshold=0.5):
        """
        Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
        The location, height, and width for pasting each mask is determined by their
        corresponding bounding boxes in boxes.

        Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

        Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
    
        Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
        """

        assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
        N = len(masks)
        if N == 0:
            return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
        if not isinstance(boxes, torch.Tensor):
            boxes = boxes.tensor
        device = boxes.device
        assert len(boxes) == N, boxes.shape

        img_h, img_w = image_shape

        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with skip_empty=True
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            # int(img_h) because shape may be tensors in tracing
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