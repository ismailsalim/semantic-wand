from pipeline.pipe import Pipeline
import cv2

class Controller:
    def __init__(self, update_canvas_cb):
        self.filename = None
        self.img = None
        self.model_results = None
        self.matte = None

        self.model = Pipeline()
        self.update_canvas_cb = update_canvas_cb

    def set_img(self, img):
        self.img = img
        self.update_canvas_cb(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def process_img(self):
        self.model_results = self.model(self.img)
        self.matte = self.model_results['mattes'][-1]
        self.update_canvas_cb(self.matte, matte=True)

        