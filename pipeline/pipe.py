# local application libraries
from pipeline.coarse_stage import CoarseStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

# system libraries
import os
from datetime import datetime

# external libraries
import numpy as np
import cv2


class Pipeline:
    def __init__(self, args):
        self.coarse_stage = CoarseStage(args.coarse_config, args.coarse_thresh)
       
        self.trimap_stage = TrimapStage(args.trimap_kernel_size,
                                        args.dilation, args.erosion)
        
        self.refinement_stage = RefinementStage(args.matting_weights)


    def process(self, args):
        img = cv2.imread(os.path.join(args.images_dir, args.img_file))

        start = datetime.now()
        coarse_preds = self.coarse_stage.pred(img)   
        instances = self.coarse_stage.get_instances(img, coarse_preds)
        subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
        end = datetime.now()
        print('Coarse stage takes:', end - start, 'seconds!')
        self.save(instances, args.img_file, 'instance_preds', args.instance_preds_dir) 
        self.save(subj.astype(int)*255, args.img_file, 'subj_pred', args.subj_masks_dir)
    

        state = datetime.now()
        trimap = self.trimap_stage.process(subj.astype(float), size)
        end = datetime.now()
        print('Trimap stage takes:', end - start, 'seconds!')
        self.save(trimap*255, args.img_file, 'trimap', args.trimaps_dir)


        start = datetime.now()
        matte = self.refinement_stage.process(trimap, img)
        end = datetime.now()
        print('Refinement stage takes:', end - start, 'seconds!')
        self.save(matte*255, args.img_file, 'matte', args.final_mattes_dir)

    
    def save(self, img, file_name, file_type, dir):
        output_file_name = '{0}_{2}{1}'.format(*os.path.splitext(file_name), file_type)
        cv2.imwrite(os.path.join(dir, output_file_name), img)

    

