from trimap_network.models import build_model, DataWrapper

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import cv2
import numpy as np
import logging
import math
import time

pipe_logger = logging.getLogger("pipeline")
device = torch.device("cuda:0") 

class TrimapStage: 
    """
    Performs automatic trimap generation using the instance probability mask (heatmap)
    output from the Masking Stage.
    """
    def __init__(self, def_fg_threshold=0.99, def_bg_threshold=0.1,
                        lr=0.001, batch_size=12000,
                        unknown_lower_bound=0.01, unknown_upper_bound=0.99, 
                        with_optimisation=True):
        
        self.def_fg_threshold = def_fg_threshold
        self.def_bg_threshold = def_bg_threshold
        self.lr = lr
        self.batch_size = batch_size
        self.unknown_lower_bound = unknown_lower_bound
        self.unknown_upper_bound = unknown_upper_bound
        self.with_optimisation = with_optimisation

        pipe_logger.info("def_fg_thresh: {}, def_bg_thresh: {}".format(
            def_fg_threshold, def_bg_threshold,
        ))
        pipe_logger.info("lr: {}, batch_size: {}".format(lr, batch_size))
        
        pipe_logger.info("unknown_lower_bound,: {}, unknown_upper_bound,: {}".format(
           unknown_lower_bound, unknown_upper_bound))
      


    def get_trimap(self, heatmap, img, bounding_box, annotated_img=None):
        """
        Obtain trimap for the image given Masking Stage output. 

        Returns:
            trimap (numpy.array):
                2D grayscale images with same (h, w) and img
        """
        pipe_logger.info("Trimap generation starting...")

        fg_mask = heatmap > self.def_fg_threshold
        bg_mask = heatmap > self.def_bg_threshold

        if annotated_img is not None:
            fg_mask = np.where(annotated_img != -1, annotated_img, fg_mask).astype(bool)
            bg_mask = np.where(annotated_img != -1, annotated_img, bg_mask).astype(bool)

        if self.with_optimisation:
            trimap = self._optimise_trimap(bounding_box, img, heatmap, fg_mask, bg_mask)

        else:
            trimap = np.zeros(img.shape[:2], dtype=float)
            trimap[fg_mask] = 1
            trimap[np.logical_and(~fg_mask, bg_mask)] = 0.5

        pipe_logger.info("Trimap generated!")
        return trimap


    def process_alpha(self, alpha, trimap, level):
        """
        Feedback definite foreground/background alpha estimation into a trimap.
        """
        trimap[alpha==0.0] = 0.0
        trimap[alpha==1.0] = 1.0
        return trimap


    def _optimise_trimap(self, bounding_box, img, heatmap, fg_mask, bg_mask):
        boundary_mask = self._expand_bounds(bounding_box, img)

        train_data, infer_data = self._preprocess_data(img, heatmap, fg_mask, bg_mask, boundary_mask)

        self.trimap_generator = build_model()

        self._train(self.trimap_generator, train_data)

        unknown_preds = self._infer(self.trimap_generator, infer_data)

        trimap = np.zeros(img.shape[:2], dtype=float) 
        trimap[fg_mask] = 1.0

        coords = np.argwhere(bg_mask != fg_mask)
        for coord, pred in zip(coords, unknown_preds):
            trimap[coord[0], coord[1]] = pred

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


    def _preprocess_data(self, img, heatmap, fg_mask, bg_mask, boundary_mask):
        X_fg = self._format_features(img, heatmap, fg_mask)
        X_bg = self._format_features(img, heatmap, np.logical_and(~bg_mask, boundary_mask))

        y_fg = np.ones(len(X_fg))
        y_bg = np.zeros(len(X_bg))

        X_train = np.vstack((X_fg, X_bg))
        y_train = np.hstack((y_fg, y_bg))
   
        train_std = X_train.std(axis=0)
        train_mean = X_train.mean(axis=0)
        X_train_norm = (X_train - train_mean)/train_std

        X_infer = self._format_features(img, heatmap, bg_mask != fg_mask)
        X_infer_norm = (X_infer - train_mean)/train_std

        X_train_final = self._add_fourier_features(X_train_norm, X_train_norm[:, -2:]) # coords
        X_infer_final = self._add_fourier_features(X_infer_norm, X_infer_norm[:, -2:]) # coords

        train_data = DataWrapper(torch.from_numpy(X_train_final).float().to(device),
                        torch.from_numpy(y_train).float().to(device))
        infer_data = DataWrapper(torch.from_numpy(X_infer_final).float().to(device))

        return train_data, infer_data
        

    def _format_features(self, img, heatmap, mask):
        colours = img[mask]
        coords = np.argwhere(mask)
        mask_probs = heatmap[mask]
        return np.hstack((mask_probs[:, np.newaxis], colours, coords))
        

    def _add_fourier_features(self, data, features):
        sin_trans= np.sin(2 * np.pi * features)
        cos_trans= np.cos(2 * np.pi * features)
        return np.hstack((data, sin_trans, cos_trans, 
                        2*sin_trans, 2*cos_trans,
                        3*sin_trans, 3*cos_trans))


    def _train(self, model, train_data):
        pipe_logger.info("Training trimap generator on new image...")
        y = train_data.y.clone().cpu().float().numpy()
        class_counts = np.unique(y, return_counts=True)[1]
        weights = 1.0 / class_counts
        samples_weight = torch.tensor([weights[int(t)] for t in y])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, sampler=sampler)

        criterion = nn.BCEWithLogitsLoss() 
        optimiser = optim.Adam(model.parameters(), lr=self.lr)  
        model.train()
        
        converging = True
        epoch = 0
        previous_loss = math.inf
        start = time.time()
        while converging:
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1)) + torch.mean(torch.log(y_pred)*y_pred) 

                epoch_loss += loss.item()

                loss.backward()
                optimiser.step()
            
            pipe_logger.info("Epoch: {} | Loss: {}".format(epoch, epoch_loss))
            
            if epoch_loss > previous_loss:
                end = time.time()
                converging = False
            else:
                previous_loss = epoch_loss
                epoch  += 1


    def _infer(self, model, infer_data):
        infer_loader = DataLoader(dataset=infer_data, batch_size=self.batch_size, shuffle=False)
        

        y_preds = np.array([], dtype=float)
        model.eval()
        with torch.no_grad():
            for X_batch in infer_loader:
                X_batch.to(device)
                y_batch_preds = model(X_batch)
                y_preds = np.hstack((y_preds, y_batch_preds.to('cpu').numpy().squeeze()))

        trimap_preds = np.where(y_preds < self.unknown_lower_bound, 0, y_preds)
        trimap_preds = np.where(np.logical_and(trimap_preds >= self.unknown_lower_bound, 
                                                trimap_preds < self.unknown_upper_bound), 
                                                0.5, trimap_preds)
        trimap_preds = np.where(trimap_preds >= self.unknown_upper_bound, 1, trimap_preds)
        return trimap_preds 
        

    