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


class Pipeline:
    def __init__(self, args):
        self.coarse_stage = CoarseStage(args.coarse_config, args.coarse_thresh)
        self.trimap_stage = TrimapStage(args.kernel_scale_factor, args.kernel_shape)
        self.refinement_stage = RefinementStage(args.matting_weights)


    def process(self, args):
        img = cv2.imread(os.path.join(args.images_dir, args.img_file))
        if img is None:
            raise ImageNotFoundError

        logging.debug('\n')
        logging.debug('Image file: {}'.format(args.img_file))
        logging.debug('Shape before resize: {}'.format(img.shape))
        
        # note detectron 2 finds optimal size anyway 
        img_size_orig = img.shape
        h, w = img.shape[:2]

        if h > args.max_dim and w > args.max_dim:
            img = self.resize(img, args.max_dim)

        logging.debug('Shape after resize: {}'.format(img.shape))

        start = time.time()
        coarse_preds = self.coarse_stage.pred(img)   
        instances = self.coarse_stage.get_instances(img, coarse_preds)
        subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
        logging.debug('Subject size is {}'.format(size))
        end = time.time()
        logging.debug('Coarse stage takes: {} seconds!'.format(end - start))
        self.save(instances, args.img_file, 'instance_preds', args.instance_preds_dir) 
        self.save(subj.astype(int)*255, args.img_file, 'subj_pred', args.subj_masks_dir)

        state = time.time()
        trimap = self.trimap_stage.process(subj.astype(float), size)
        end = time.time()
        logging.debug('Trimap stage takes: {} seconds'.format(end - start))
        self.save(trimap*255, args.img_file, 'trimap', args.trimaps_dir)


        start = time.time()
        matte = self.refinement_stage.process(trimap, img, img_size_orig)
        end = time.time()
        logging.debug('Refinement stage takes: {} seconds!'.format(end - start))
        self.save(matte*255, args.img_file, 'matte', args.final_mattes_dir)
        
    
    def resize(self, img, dim_max=1000):
        (h, w) = img.shape[:2]
        if h > w:
            r = dim_max/float(h)
            dim = (int(w*r), dim_max)
        else:
            r = dim_max/float(w)
            dim = (dim_max, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    
    def save(self, img, file_name, output_type, dir):
        output_file_name = '{0}_{1}{2}'.format(os.path.splitext(file_name)[0], 
                                                output_type, '.png')
        cv2.imwrite(os.path.join(dir, output_file_name), img)


class ImageNotFoundError(Exception):
    """Thrown when image file isn't/can't be read"""
    def __init__(self):
        msg = 'Image not found!'
        super(ImageNotFoundError, self).__init__(msg)
    
