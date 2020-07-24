import cv2
import numpy as np

class Controller:
    def __init__(self, model, update_canvas_cb):
        self.filename = None
        self.img = None
        self.model_results = None
        self.trimap = None
        self.fg = None
        self.alpha = None
        self.matte = None


        self.model = model
        self.update_canvas_cb = update_canvas_cb


    def set_img(self, img):
        self.img = img
        self.update_canvas_cb(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))


    def process_img(self, annotations):
        # annotations is PIL.Image.Image
        annotations = self.preprocess_annotations(annotations)
        self.model_results = self.model(self.img, annotations)
        self.trimap = self.model_results['trimaps'][-1] # get final trimap
        self.fg = self.model_results['foregrounds'][-1] # get final fg
        self.alpha = self.model_results['alphas'][-1] # get final alpha
        self.matte = self.model_results['mattes'][-1] # get final matte
        self.update_canvas_cb(self.matte, with_matte=True)


    def preprocess_annotations(self, annotations):
        annotations = np.array(annotations, dtype=np.int32)
        annotations[annotations == 128] = -1 # convert unnatoated pixels
        annotations[annotations == 255] = 1 # convert fg annotations
        annotations[annotations == 0] = 0 # convert bg annotations
        return annotations

        