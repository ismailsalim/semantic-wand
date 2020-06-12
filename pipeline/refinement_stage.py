# local application libraries
from matting.networks.models import build_model
from matting.demo import np_to_torch, pred, scale_input

# external libraries
import numpy as np

class RefinementStage:
    def __init__(self):
        self.model = build_model('resnet50_GN_WS', 'fba_decoder', './matting/FBA.pth')


    def process(self, trimap, img):
        h, w = trimap.shape

        # model requires two channel trimap
        fba_trimap = np.zeros((h, w, 2)) 
        fba_trimap[trimap==1, 1] = 1
        fba_trimap[trimap==0, 0] = 1
        
        img = img/255.0
    
        fg, bg, alpha = pred(img, fba_trimap, self.model)
        return fg, bg, alpha
