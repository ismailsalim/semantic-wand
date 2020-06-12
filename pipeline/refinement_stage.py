# local application libraries
from matting.networks.models import build_model
from matting.networks.transforms import trimap_transform, groupnorm_normalise_image

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
        

    def pred(self, img, trimap, model):
        h, w = trimap.shape[:2]

        scaled_img = self.scale_input(img, 1.0)
        scaled_trimap = self.scale_input(trimap, 1.0)

        with torch.no_grad():
            image_torch = self.to_tensor(scaled_img)
            trimap_torch = self.to_tensor(scaled_trimap)

            trimap_transformed_torch = self.to_tensor(trimap_transform(scaled_trimap))
            image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

            output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

            output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
        
        fg = output[:, :, 1:4]
        bg = output[:, :, 4:7]
        alpha = output[:, :, 0]

        fg[alpha == 1] = img[alpha == 1]
        bg[alpha == 0] = img[alpha == 0]
        alpha[trimap[:, :, 0] == 1] = 0
        alpha[trimap[:, :, 1] == 1] = 1
        
        return fg, bg, alpha


    def to_tensor(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


    def scale_input(self, x, scale):
        h, w = x.shape[:2]
        h1 = int(np.ceil(scale * h / 8) * 8)
        w1 = int(np.ceil(scale * w / 8) * 8)
        x_scale = cv2.resize(x, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        return x_scale


