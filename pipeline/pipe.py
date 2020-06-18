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
        self.img = cv2.imread(os.path.join(args.images_dir, self.input_file)) 
        if self.img is None:
            raise ImageNotFoundError
        
        self.max_img_dim = args.max_img_dim # image size

        self.iterations = args.iterations # trimap/alpha feedback loops

        # configure directories for saving results
        self.instances = args.instance_preds_dir
        self.subjs = args.subj_masks_dir
        self.trimaps = args.trimaps_dir
        self.fgs = args.fgs_dir
        self.alphas = args.alphas_dir
        self.mattes = args.final_mattes_dir

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

        # trimap = self.to_trimap_stage(subj, size)
        # self.to_refinement_stage(trimap, self.img)  

        self.to_matting_loop(subj, size, 1)

    
    def to_coarse_stage(self):
        start = time.time()

        coarse_preds = self.coarse_stage.pred(self.img)   
        instances = self.coarse_stage.get_instances(self.img, coarse_preds)
        subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
        logging.debug('Subject size is {} pixels'.format(size))

        end = time.time()
        logging.debug('Coarse stage takes: {} seconds!'.format(end - start))
        
        self.save(instances, 'instance_preds', self.instances) 
        self.save(subj*255, 'subj_pred', self.subjs)

        return subj, size
    
    
    def to_matting_loop(self, coarse_mask, size, iteration):
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

        self.save(trimap*255, 'trimap_{}'.format(iteration), self.trimaps)
        return trimap
    

    def to_refinement_stage(self, trimap, img, iteration):
        start = time.time()
        
        fg, alpha = self.refinement_stage.process(trimap, img)         
        matte = cv2.cvtColor(fg, cv2.COLOR_RGB2RGBA) 
        matte[:, :, 3] = alpha
        
        end = time.time()
        logging.debug('Refinement stage takes: {} seconds!'.format(end - start))

        self.save(alpha*255, 'alpha_{}'.format(iteration), self.alphas)
        self.save(fg*255, 'foreground_{}'.format(iteration), self.fgs)
        self.save(matte*255, 'matte_{}'.format(iteration), self.mattes)  

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

    
    def save(self, img, output_type, to_dir):
        output_file_name = '{0}_{1}{2}'.format(os.path.splitext(self.input_file)[0], 
                                               output_type, '.png')
        cv2.imwrite(os.path.join(to_dir, output_file_name), img)


