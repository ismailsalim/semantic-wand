# local application libraries
from matting.networks.models import build_model
from matting.networks.transforms import trimap_transform, groupnorm_normalise_image
from matting.dataloader import PredDataset

# system libraries
import os
import argparse

# external libraries
import cv2
import numpy as np
import torch

class RefinementStage:
    def __init__(self, weights):
        self.model = build_model('resnet50_GN_WS', 'fba_decoder', weights)


    def process(self, trimap, img):
        h, w = trimap.shape

        # model requires two channel trimap
        fba_trimap = np.zeros((h, w, 2)) 
        fba_trimap[trimap==1, 1] = 1
        fba_trimap[trimap==0, 0] = 1
        
        img = img/255.0
    
        fg, bg, alpha = self.pred(img, fba_trimap, self.model)
        return fg, bg, alpha
        

    def pred(self, image_np, trimap_np, model):
        h, w = trimap_np.shape[:2]

        image_scale_np = self.scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
        trimap_scale_np = self.scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

        with torch.no_grad():
            image_torch = self.np_to_torch(image_scale_np)
            trimap_torch = self.np_to_torch(trimap_scale_np)

            trimap_transformed_torch = self.np_to_torch(trimap_transform(trimap_scale_np))
            image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

            output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

            output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
        
        fg = output[:, :, 1:4]
        bg = output[:, :, 4:7]
        alpha = output[:, :, 0]

        fg[alpha == 1] = image_np[alpha == 1]
        bg[alpha == 0] = image_np[alpha == 0]
        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1
        
        return fg, bg, alpha


    def np_to_torch(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


    def scale_input(self, x, scale, scale_type):
        h, w = x.shape[:2]
        h1 = int(np.ceil(scale * h / 8) * 8)
        w1 = int(np.ceil(scale * w / 8) * 8)
        x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
        return x_scale


