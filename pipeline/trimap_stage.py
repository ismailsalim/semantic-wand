import numpy as np
import detectron2
from detectron2.layers.mask_ops import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom

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

        trimap = np.zeros(fg_mask.shape, dtype='float64') 
        trimap[fg_mask == 1.0] = 1.0
        trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5

        if annotated_img is not None:
            trimap[annotated_img == 1] = 1.0
            trimap[annotated_img == 0] = 0.0

        return trimap, fg_thresh, unknown_thresh


    def process_alpha(self, alpha, trimap, level):
        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        return trimap


    def _get_mask(self, subject, thresholds, preference, target=None):
        if target is not None:
            masks = [self._generate_mask(subject, thresh) for thresh in thresholds]
            matching = [np.sum(np.logical_and(mask==True, target==True)) for mask in masks]
            optimal_idx = matching.index(preference(matching))
            return masks[optimal_idx], thresholds[optimal_idx] 
        else:
            optimal_mask = self._generate_mask(subject, preference(thresholds))
            return optimal_mask, preference(thresholds)


    def _generate_mask(self, subject, thresh):
        binary_mask = retry_if_cuda_oom(paste_masks_in_image)(
            subject.pred_masks[:, 0, :, :],  # N, 1, M, M
            subject.pred_boxes,
            subject.image_size,
            thresh
        )
        return binary_mask.cpu().numpy().squeeze()


