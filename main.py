from pipeline.pipe import Pipeline
from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

from demo.app import App

from utils.logger import setup_logger

import argparse
import os
import time
import tkinter as tk
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    usage = parser.add_mutually_exclusive_group()
    usage.add_argument('-interactive', action='store_true', help='To run interactive demo')
    usage.add_argument('-intermediate', action='store_true', 
                        help='One image and scribbles with complete intermediate results')
    usage.add_argument('-multiple', action='store_true',
                        help='Folder of images and scribbles with simple results')

    # USE CASE 1: One image with complete intermediate and final results 
    parser.add_argument('--image',
                        help='Path to specific image to be processed')
    parser.add_argument('--scribbles', 
                        help='Name of a specific annotated image (inc. extension)')

    # USE CASE 2: Folder of images and scribbles with only final results
    parser.add_argument('--images_dir', help='Path to folder of images that will be processed')
    parser.add_argument('--scribbles_dir', help='Path to folder of scribbles that will be used with images')
    parser.add_argument('--fgs_dir', help='Path to folder where alpha output will be saved')
    parser.add_argument('--alphas_dir', help='Path to folder where fg output will be saved')
    parser.add_argument('--mattes_dir', help='Path to folder where matte output will be saved')

    # for pipeline specification
    parser.add_argument('--max_img_dim', type=int, default=800, 
                        help='Number of pixels that the image\'s maximum dimension is scaled to for processing')
    parser.add_argument('--feedback_thresh', type=int, default=0.01, 
                         help='Min proportional change in trimap\'s def area to pass back into refinement stage')
    # for masking stage specification
    parser.add_argument('--mask_config', default='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        help='YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)')
    parser.add_argument('--instance_thresh', type=float, default=0.05, 
                        help='Mask R-CNN score threshold for instance recognition')
   
    # for trimap stage specification
    parser.add_argument('--def_fg_thresh', type=float, default=0.99,
                        help='Threshold above which mask pixels labelled as def fg for trimap network training')
    parser.add_argument('--unknown_thresh', type=float, default=0.1,
                        help='Threshold below which mask pixels labelled as def bg for trimap network training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate during training of trimap network')
    parser.add_argument('--batch_size', type=int, default=12000, 
                        help='Batch size used for training of trimap network')    
    parser.add_argument('--unknown_lower_bound', type=float, default=0.01,
                        help='Probability below which trimap network inference is classified as bg')
    parser.add_argument('--unknown_upper_bound', type=float, default=0.99,
                        help='Probability above which trimap network inference is classified as fg')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Path to pre-trained matting model')

    args = parser.parse_args()
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    pipe_logger = setup_logger("pipeline", "logs/pipeline.log")
    pipe_logger.info("\n\n INITIALISING PIPELINE \n")

    masking_stage = MaskingStage(args.mask_config, args.instance_thresh)
    trimap_stage = TrimapStage(args.def_fg_thresh, args.unknown_thresh, args.lr, args.batch_size,
                                args.unknown_lower_bound, args.unknown_upper_bound)  
    refinement_stage = RefinementStage(args.matting_weights)
    
    pipeline = Pipeline(masking_stage, trimap_stage, refinement_stage, 
                        args.feedback_thresh, args.max_img_dim) 
    
    if args.multiple:
        assert None not in (args.images_dir, args.scribbles_dir, args.fgs_dir, args.alphas_dir, args.mattes_dir), (
            "Input and output directories must be specified with this usage!"
        )
        
        img_ids = sorted(os.listdir(args.images_dir))
        scribble_ids = sorted(os.listdir(args.scribbles_dir))

        assert len(img_ids) == len(scribble_ids), "Image and scribble folders must be same size!"
           
        if not os.path.exists(args.fgs_dir):
            os.mkdir(args.fgs_dir)

        if not os.path.exists(args.alphas_dir):
            os.mkdir(args.alphas_dir)

        if not os.path.exists(args.mattes_dir):
            os.mkdir(args.mattes_dir)

        print(args.images_dir)

        for img_file, scribbles_file in zip(img_ids, scribble_ids):
            
            img = cv2.imread(os.path.join(args.images_dir, img_file))
            scribbles = cv2.imread(os.path.join(args.scribbles_dir, scribbles_file), 0).astype(np.int32)
            assert img.shape[:2] == scribbles.shape[:2], (
                "Image ({}): {} and Scribbles ({}): {} must be same size!".format(img.shape[:2], img_file, 
                                                                                    scribble_file, scribbles.shape[:2]))

            scribbles[scribbles == 128] = -1 # convert unnannotated pixels
            scribbles[scribbles == 255] = 1 # convert fg scribbles
            scribbles[scribbles == 0] = 0 # convert bg scribbles

            results = pipeline(img, scribbles)

            cv2.imwrite(os.path.join(args.fgs_dir, img_file), results['foregrounds'][-1])
            cv2.imwrite(os.path.join(args.alphas_dir, img_file), results['alphas'][-1])
            cv2.imwrite(os.path.join(args.mattes_dir, img_file), results['mattes'][-1])

    if args.intermediate:
        assert None not in (args.image, args.scribbles), "Must specify one image and one corresponding scribbles"


        img = cv2.imread(args.image)
        scribbles = cv2.imread(args.scribbles, 0).astype(np.int32)
        assert img.shape[:2] == scribbles.shape[:2], (
            "Image ({}): {} and Scribbles ({}): {} must be same size!".format(img.shape[:2], os.path.splitext(os.path.basename(args.image)), 
                                                                                os.path.splitext(os.path.basename(args.scribbles)), scribbles.shape[:2]))

        scribbles[scribbles == 128] = -1 # convert unnannotated pixels
        scribbles[scribbles == 255] = 1 # convert fg scribbles
        scribbles[scribbles == 0] = 0 # convert bg scribbles

        results = pipeline(img, scribbles)

        img_id = os.path.splitext(os.path.basename(args.image))
        parent = os.path.split(args.image)[0]
        output_dir = os.path.join(parent, img_id + '_' + time.strftime("%d-%m--%H-%M-%S"))
        os.mkdir(output_dir)

        save_all_results(results, img_id, output_dir)

    if args.interactive:
        root = tk.Tk()
        app = App(root, pipeline, args.max_img_dim)
        root.deiconify()
        app.mainloop()   


def save_all_results(results, img_id, to_dir):
    for i, (img_type, pred) in enumerate(results.items()):
        save_results_type(pred, img_id, '{}_{}'.format(i, img_type), to_dir)


def save_results_type(pred, img_id, output_type, to_dir):
    if type(pred) == list: # trimaps, foregrounds, alphas, mattes
        for i, img in enumerate(pred): 
            output_file_name = '{0}_{1}_iter{2}{3}'.format(img_id, output_type, i, '.png')
            cv2.imwrite(os.path.join(to_dir, output_file_name), img)
    else: # instances, unknown_mask, fg_mask
        output_file_name = '{0}_{1}{2}'.format(img_id, output_type, '.png')
        cv2.imwrite(os.path.join(to_dir, output_file_name), pred)


if __name__ == '__main__':
    main()
