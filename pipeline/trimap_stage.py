from trimap_network.models import build_model, DataWrapper

import numpy as np
import detectron2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.utils.memory import retry_if_cuda_oom
import logging

import cv2

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit
device = torch.device("cuda:0")

logging.basicConfig(filename='trimap.log', level=logging.DEBUG, 
                format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.debug("\n")

class TrimapStage: 
    def __init__(self, def_fg_thresholds, unknown_thresholds,
                epochs, lr, batch_size, unknown_lower_bound, unknown_upper_bound):
        self.def_fg_thresholds = def_fg_thresholds
        self.unknown_thresholds = unknown_thresholds
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.unknown_lower_bound = unknown_lower_bound
        self.unknown_upper_bound = unknown_upper_bound

        logging.debug("lr: {}, epochs: {}, batch_size: {}, lower_b: {}, upper_b: {}".format(
            lr, epochs, batch_size, unknown_lower_bound, unknown_upper_bound
        ))


    def process_subject(self, subject, img, bounding_box, annotated_img=None):
        if annotated_img is not None:
            fg_mask, fg_thresh = self._get_mask(subject, self.def_fg_thresholds, min, annotated_img==0)
            unknown_mask, unknown_thresh = self._get_mask(subject, self.unknown_thresholds, max, annotated_img==1)
        else:
            fg_mask, fg_thresh = self._get_mask(subject, self.def_fg_thresholds, min)
            unknown_mask, unknown_thresh = self._get_mask(subject, self.unknown_thresholds, max)

        heatmap, _ = self._get_mask(subject)

        boundary_mask = self._expand_bounds(bounding_box, img)

        # old method
        # trimap = np.zeros(fg_mask.shape, dtype='float64') 
        # trimap[fg_mask == 1.0] = 1.0
        # trimap[np.logical_and(unknown_mask==1.0, fg_mask==0.0)] = 0.5

        # if annotated_img is not None:
        #     trimap[annotated_img == 1] = 1.0
        #     trimap[annotated_img == 0] = 0.0
        
        train_data, infer_data = self._preprocess_data(img, heatmap, fg_mask, unknown_mask, 
                                                        boundary_mask, annotated_img)
                
        # train_data, infer_data = self._preprocess_data(img, heatmap, fg_mask, unknown_mask, annotated_img)
        self.trimap_generator = build_model()
        self._train(self.trimap_generator, train_data)
        unknown_preds = self._infer(self.trimap_generator, infer_data)
        
        trimap = np.zeros(img.shape[:2], dtype=float) 
        trimap[fg_mask] = 1
        # cv2.imwrite("./trimap_fg_mask.png", trimap*255)

        coords = np.argwhere(unknown_mask != fg_mask)

        for coord, pred in zip(coords, unknown_preds):
            trimap[coord[0], coord[1]] = pred

        return heatmap, trimap, fg_mask, unknown_mask

    def process_alpha(self, alpha, trimap, level):
        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        return trimap

    def _expand_bounds(self, bounding_box, img):
        height = bounding_box[3] - bounding_box[1]
        width = bounding_box[2] - bounding_box[0]
        y_bounds = [int(bounding_box[1]-(1/5 * height)), int(bounding_box[3]+(1/5 * height))]
        x_bounds = [int(bounding_box[0]-(1/5 * width)), int(bounding_box[2]+(1/5 * width))]
        
        if x_bounds[0] < 0:
            x_bounds[0] = 0
        if x_bounds[1] > img.shape[1]:
            x_bounds[1] = img.shape[1]
        if y_bounds[0] < 0:
            y_bounds[0] = 0
        if y_bounds[1] > img.shape[0]:
            y_bounds[1] = img.shape[0]

        mask = np.ones(img.shape[:2]).astype(bool)
        mask[:y_bounds[0],:] = False
        mask[y_bounds[1]:,:] = False
        mask[:,:x_bounds[0]] = False
        mask[:,x_bounds[1]:] = False
        return mask

    def _preprocess_data(self, img, heatmap, fg_mask, unknown_mask, boundary_mask, annotated_img=None):
        if annotated_img is not None:
            fg_mask = np.where(annotated_img != -1, annotated_img, fg_mask).astype(bool)
            unknown_mask = np.where(annotated_img != -1, annotated_img, unknown_mask).astype(bool)

        X_fg = self._format_features(img, heatmap, fg_mask, annotated_img)
        X_bg = self._format_features(img, heatmap, np.logical_and(~unknown_mask, boundary_mask), annotated_img)
        # X_bg = self._format_features(img, heatmap, ~unknown_mask, annotated_img)

        
        cv2.imwrite("./fg_mask.png", fg_mask*255)
        cv2.imwrite("./bg_mask.png", np.logical_and(~unknown_mask, boundary_mask)*255)
        # cv2.imwrite("./bg_mask.png", ~unknown_mask*255)

        y_fg = np.ones(len(X_fg))
        y_bg = np.zeros(len(X_bg))

        X_train = np.vstack((X_fg, X_bg))
        y_train = np.hstack((y_fg, y_bg))
        logging.debug("Distribution of training samples {}".format(np.unique(y_train, 
                                                                return_counts=True)))
        train_std = X_train.std(axis=0)
        train_mean = X_train.mean(axis=0)
        X_train_norm = (X_train - train_mean)/train_std

        # shuffler = np.random.permutation(len(y_train))
        # X_train = X_train[shuffler]
        # y_train = y_train[shuffler]

        train_data = DataWrapper(torch.from_numpy(X_train_norm).float().to(device),
                                torch.from_numpy(y_train).float().to(device))


        X_infer = self._format_features(img, heatmap, unknown_mask != fg_mask, annotated_img)
        X_infer_norm = (X_infer - train_mean)/train_std
        cv2.imwrite("./infer_mask.png", (unknown_mask != fg_mask)*255)

        infer_data = DataWrapper(torch.from_numpy(X_infer_norm).float().to(device))
        
        return train_data, infer_data
        

    def _format_features(self, img, heatmap, mask, annotated_img=None):
        colours = img[mask]
        mask_probs = heatmap[mask]
        coords = np.argwhere(mask)
        return np.hstack((mask_probs[:, np.newaxis], colours, coords))


    def _train(self, model, train_data):
        y = train_data.y.clone().cpu().float().numpy()
        class_counts = np.unique(y, return_counts=True)[1]
        weights = 1.0 / class_counts
        samples_weight = torch.tensor([weights[int(t)] for t in y])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, sampler=sampler)

        criterion = nn.BCEWithLogitsLoss()
        optimiser = optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for e in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1))
                epoch_loss += loss.item()

                loss.backward()
                optimiser.step()

            logging.debug("Epoch: {} | Loss: {}".format(e, epoch_loss))


    def _infer(self, model, infer_data):
        infer_loader = DataLoader(dataset=infer_data, batch_size=self.batch_size, shuffle=False)
        
        y_preds = np.array([], dtype=float)
        model.eval()
        with torch.no_grad():
            for X_batch in infer_loader:
                X_batch.to(device)
                y_batch_preds = model(X_batch)
                y_preds = np.hstack((y_preds, y_batch_preds.to('cpu').numpy().squeeze()))

        # (optimise) thresholds used for trimap
        trimap_preds = np.where(y_preds < self.unknown_lower_bound, 0, y_preds)
        trimap_preds = np.where(np.logical_and(trimap_preds >= self.unknown_lower_bound, 
                                                trimap_preds < self.unknown_upper_bound), 
                                                0.5, trimap_preds)
        trimap_preds = np.where(trimap_preds >= self.unknown_upper_bound, 1, trimap_preds)
        logging.debug("Preds: {}".format(np.unique(trimap_preds, return_counts=True)))
        return trimap_preds


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