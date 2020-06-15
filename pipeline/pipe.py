# local application libraries
from pipeline.coarse_stage import CoarseStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

# system libraries
import os
from datetime import datetime
import logging
logging.basicConfig(filename='trimap.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

# external libraries
import numpy as np
import cv2


class Pipeline:
    def __init__(self, args):
        self.coarse_stage = CoarseStage(args.coarse_config, args.coarse_thresh)
        self.trimap_stage = TrimapStage(args.kernel_scale_factor, args.kernel_shape)
        self.refinement_stage = RefinementStage(args.matting_weights)


    def process(self, args):
        img = cv2.imread(os.path.join(args.images_dir, args.img_file))
        logging.debug('\n')
        logging.debug('Image file: {}'.format(args.img_file))
        logging.debug('Image shape: {}'.format(img.shape))

        start = datetime.now()
        coarse_preds = self.coarse_stage.pred(img)   
        instances = self.coarse_stage.get_instances(img, coarse_preds)
        subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
        end = datetime.now()
        logging.debug('Coarse stage takes: {} seconds!'.format(end - start))
        self.save(instances, args.img_file, 'instance_preds', args.instance_preds_dir) 
        self.save(subj.astype(int)*255, args.img_file, 'subj_pred', args.subj_masks_dir)
    

        state = datetime.now()
        trimap = self.trimap_stage.process(subj.astype(float), size)
        end = datetime.now()
        logging.debug('Trimap stage takes: {} seconds'.format(end - start))
        self.save(trimap*255, args.img_file, 'trimap', args.trimaps_dir)


        # start = datetime.now()
        # matte = self.refinement_stage.process(trimap, img)
        # end = datetime.now()
        # print('Refinement stage takes:', end - start, 'seconds!')
        # self.save(matte*255, args.img_file, 'matte', args.final_mattes_dir)

    
    def resize(self, img, dim_max=600):
        (h, w) = img.shape[:2]
        if h > w:
            r = dim_max/float(h)
            dim = (int(w*r), dim_max)
        else:
            r = dim_max/float(w)
            dim = (dim_max, int(h*r))

        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    
    def save(self, img, file_name, file_type, dir):
        output_file_name = '{0}_{2}{1}'.format(*os.path.splitext(file_name), file_type)
        cv2.imwrite(os.path.join(dir, output_file_name), img)

    
