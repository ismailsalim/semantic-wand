# local application libraries
from pipeline.coarse_stage import CoarseStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

# system libraries
import os
import time
import logging
logging.basicConfig(filename='pipeline.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

# external libraries
import torch
import numpy as np
import cv2


class ImageNotFoundError(Exception):
    """Thrown when image file isn't/can't be read"""
    def __init__(self):
        msg = 'Image specified not found!'
        super(ImageNotFoundError, self).__init__(msg)
    

class Pipeline:
    def __init__(self, args):
        self.input_file = args.img_filename
        self.img_id = os.path.splitext(self.input_file)[0]
        
        if args.img_dir is None:
            self.img_dir = os.path.join('./examples', self.img_id)
        else:
            self.img_dir = args.img_dir

        self.img = cv2.imread(os.path.join(self.img_dir, self.input_file))
        if self.img is None:
            raise ImageNotFoundError
        
        self.output_dir = os.path.join(self.img_dir, time.strftime("%d-%m--%H-%M-%S"))
        os.mkdir(self.output_dir)

        self.max_img_dim = args.max_img_dim # image size fed into pipeline

        self.iterations = args.iterations # trimap/alpha feedback loops

        # instantiate pipeline stages
        self.coarse_stage = CoarseStage(args.coarse_config, args.coarse_thresh)
        self.trimap_stage = TrimapStage(args.kernel_scale_factor, args.kernel_shape, self.iterations)
        self.refinement_stage = RefinementStage(args.matting_weights)


    def process(self):
        logging.debug('\nImage file: {}'.format(self.input_file))
        logging.debug('Image shape: {}'.format(self.img.shape))
        
        # rescale according to maximum image dimension specified
        h, w = self.img.shape[:2]
        if h > self.max_img_dim or w > self.max_img_dim:
            self.img = self.rescale(self.img)
            logging.debug('Resizing to: {}'.format(self.img.shape))

        subj, size = self.to_coarse_stage()
        
        self.to_matting_loop(subj, size, 1)

    
    def to_coarse_stage(self):
        start = time.time()

        coarse_preds = self.coarse_stage.pred(self.img)   
        instances = self.coarse_stage.get_instances(self.img, coarse_preds)
        subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
        logging.debug('Subject size is {} pixels'.format(size))

        end = time.time()
        logging.debug('Coarse stage takes: {} seconds!'.format(end - start))
        
        self.save(instances, '1_instance_preds') 
        self.save(subj*255, '2_subj_pred')

        return subj, size
    
    
    def to_matting_loop(self, coarse_mask, size, iteration):
        logging.debug("Starting matting loop iteration 1...")
        trimap = self.to_trimap_stage(coarse_mask, size, iteration)
        fg, alpha, matte = self.to_refinement_stage(trimap, self.img, iteration)
        
        if iteration < self.iterations:
            self.to_matting_loop(alpha, size, iteration+1)

        return fg, alpha, matte


    def to_trimap_stage(self, subj, size, iteration):
        start = time.time()
        
        trimap = self.trimap_stage.process(subj, size, iteration)
        
        end = time.time()
        logging.debug('Trimap stage takes: {} seconds'.format(end - start))

        self.save(trimap*255, '3_trimap_iter_{}'.format(iteration))
        return trimap
    

    def to_refinement_stage(self, trimap, img, iteration):
        start = time.time()
        
        fg, alpha = self.refinement_stage.process(trimap, img)         
        matte = cv2.cvtColor(fg, cv2.COLOR_RGB2RGBA) 
        matte[:, :, 3] = alpha
        
        end = time.time()
        logging.debug('Refinement stage takes: {} seconds!'.format(end - start))

        self.save(alpha*255, '4_alpha_iter{}'.format(iteration))
        self.save(fg*255, '5_foreground__iter{}'.format(iteration))
        self.save(matte*255, '6_matte_iter{}'.format(iteration))  

        return fg, alpha, matte


    def rescale(self, img):
        (h, w) = img.shape[:2]

        if h > w:
            r = self.max_img_dim/float(h)
            dim = (int(w*r), self.max_img_dim)
        else:
            r = self.max_img_dim/float(w)
            dim = (self.max_img_dim, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    
    def save(self, img, output_type):
        output_file_name = '{0}_{1}{2}'.format(self.img_id, output_type, '.png')
        cv2.imwrite(os.path.join(self.output_dir, output_file_name), img)


