# local application libraries
from pipeline.coarse_stage import CoarseStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

# system libraries
import os

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
        images, filenames = self.load_images(args.images_dir)
        
        for img, filename in zip(images, filenames):
            coarse_preds = self.coarse_stage.pred(img)
            
            instances = self.coarse_stage.get_instances(img, coarse_preds)
            self.save(instances, filename, 'instance_preds', args.instance_preds_dir)
            
            subj, size = self.coarse_stage.get_subj_mask(coarse_preds)
            self.save(subj.astype(int)*255, filename, 'subj_pred', args.subj_masks_dir)

            trimap = self.trimap_stage.process(subj.astype(float), size)
            self.save(trimap*255, filename, 'trimap', args.trimaps_dir)

            matte = self.refinement_stage.process(trimap, img)
            self.save(matte*255, filename, 'matte', args.final_mattes_dir)


    def load_images(self, dir):
        images = []
        filenames = []
        
        for filename in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, filename))
            images.append(img)
            filenames.append(filename)
        
        return images, filenames   

    
    def save(self, img, file_name, file_type, dir):
        output_file_name = '{0}_{2}{1}'.format(*os.path.splitext(file_name), file_type)
        cv2.imwrite(os.path.join(dir, output_file_name), img)

    

