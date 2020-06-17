###############################################################################
# Original source code for the FBA Matting model used in the refinement stage 
# can be found at https://github.com/MarcoForte/FBA_Matting
#
# Pre-trained model weights can also be found at the url above. Note that these
# are covered by the Deep Image Matting Dataset License Agreement for which
# the reader should refer to https://sites.google.com/view/deepimagematting
#
# This code leverages the FBA matting model trained with GroupNorm and Weight
# Standardisation specifically (filenames are preserved as per the source)
###############################################################################

# local application libraries
from matting_network.models import build_model

# system libraries
import os
import argparse

# external libraries
import cv2
import numpy as np
import torch


class RefinementStage:
    def __init__(self, weights):
        self.model = build_model(weights) # MattingModule


    def process(self, trimap, img):
        h, w = trimap.shape

        # fba matting network requires two channel trimap
        fba_trimap = np.zeros((h, w, 2)) 
        fba_trimap[trimap==1, 1] = 1
        fba_trimap[trimap==0, 0] = 1
        
        img = img/255.0
    
        fg, alpha = self.pred(img, fba_trimap, self.model)
        
        matte = fg*alpha[:, :, None]
        matte = cv2.cvtColor(matte, cv2.COLOR_RGB2RGBA) 
        matte[:, :, 3] = alpha

        return fg, alpha, matte
        

    def pred(self, img, trimap, model):
        h, w = img.shape[:2]

        img_scaled = self.scale_input(img, 1.0)
        trimap_scaled = self.scale_input(trimap, 1.0)

        with torch.no_grad():
            img_torch = self.np_to_torch(img_scaled)
            trimap_torch = self.np_to_torch(trimap_scaled)

            img_trans_torch = self.normalise_img(img_torch.clone())
            trimap_trans_torch = self.np_to_torch(self.blur_trimap(trimap_scaled))

            output = model(img_torch, trimap_torch, img_trans_torch, trimap_trans_torch)    
            output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
        
        #  only using 4 out of the 7 output channels 
        alpha = output[:, :, 0]
        fg = output[:, :, 1:4]
        
        fg[alpha == 1] = img[alpha == 1]
        alpha[trimap[:, :, 0] == 1] = 0
        alpha[trimap[:, :, 1] == 1] = 1

        return fg, alpha


    def scale_input(self, x, scale):
        h, w = x.shape[:2]
        h1 = int(np.ceil(scale * h / 8) * 8)
        w1 = int(np.ceil(scale * w / 8) * 8)
        x_scale = cv2.resize(x, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        return x_scale


    def np_to_torch(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
        

    def blur_trimap(self, trimap):
        h, w = trimap.shape[0], trimap.shape[1]

        # gaussian blurring at 3 different scales of definite fg and bg
        clicks = np.zeros((h, w, 6))
        for k in range(2):
            if(np.count_nonzero(trimap[:, :, k]) > 0):
                dt_mask = -self.distance_transform(1-trimap[:, :, k])**2
                L = 320
                clicks[:, :, 3*k] = np.exp(dt_mask/(2*((0.02*L)**2)))
                clicks[:, :, 3*k+1] = np.exp(dt_mask/(2*((0.08*L)**2)))
                clicks[:, :, 3*k+2] = np.exp(dt_mask/(2*((0.16*L)**2)))

        return clicks


    def distance_transform(self, trimap_channel):
        return cv2.distanceTransform((trimap_channel*255).astype(np.uint8), cv2.DIST_L2, 0)


    def normalise_img(self, img):
        # applying group normalisation
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        
        for i in range(3):
            img[..., i, :, :] = (img[..., i, :, :]-mean[i])/std[i]  

        return img

