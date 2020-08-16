from utils.preprocess_image import preprocess_scribbles

import cv2
import numpy as np

class Controller:
    def __init__(self, model, update_canvas_cb):
        self.filename = None
        self.img = None

        self.model_results = None

        self.model = model
        self.update_canvas_cb = update_canvas_cb


    def set_img(self, img):
        self.img = img

        self.instances = None
        self.heatmap = None
        self.trimap = None
        self.alpha = None
        self.matte = None
        
        self.update_canvas_cb(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))


    def process_img(self, scribbles):
        scribbles = preprocess_scribbles(scribbles)
        self.model_results = self.model(self.img, scribbles)
        self.instances = self.model_results['instances']
        self.heatmap = self.model_results['heatmap']
        self.trimap = self.model_results['trimaps'][-1] # get final trimap
        self.alpha = self.model_results['alphas'][-1] # get final alpha
        self.matte = self.model_results['mattes'][-1] # get final matte
        self.update_canvas_cb(self.matte, with_matte=True)


        