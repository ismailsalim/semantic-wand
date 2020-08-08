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
    args = parse_args()
    
    pipe_logger = setup_logging(args)

    pipe_logger.info("\n\n INITIALISING PIPELINE...")

    masking_stage = MaskingStage(args.mask_config, args.instance_thresh)
    
    trimap_stage = TrimapStage(args.def_fg_thresh, args.unknown_thresh, args.lr, 
                                args.batch_size, args.unknown_lower_bound, args.unknown_upper_bound,
                                args.no_optimisation)  
    
    refinement_stage = RefinementStage(args.matting_weights)
    
    pipeline = Pipeline(masking_stage, trimap_stage, refinement_stage, args.feedback_thresh, args.max_img_dim) 
    
    if args.interactive: # use case 1
        root = tk.Tk()
        app = App(root, pipeline, args.max_img_dim)
        root.deiconify()
        app.mainloop()   

    if args.intermediate:  # use case 2
        process_image_intermediate(args, pipeline)

    if args.eval: # use case 3
        process_images_eval(args, pipeline)


def parse_args():
    parser = argparse.ArgumentParser()

    usage = parser.add_mutually_exclusive_group(required=True)
    usage.add_argument("-interactive", action="store_true", help="To run interactive demo")
    usage.add_argument("-intermediate", action="store_true", 
                        help="To process one image and corresponding scribbles with complete intermediate results")
    usage.add_argument("-eval", action="store_true",
                        help="To process a folder of images and scribbles with selected final results required for eval")

    # logging set up
    parser.add_argument("--no_logs", action="store_true", help="To disable logging")

    # USE CASE 2: One image with complete intermediate and final results 
    parser.add_argument("--image", type=str, help="Path to a specific image to be processed")
    parser.add_argument("--scribbles", type=str, help="Paths to a specific scribbles image")

    # USE CASE 3: Folder of images and scribbles with only final results
    parser.add_argument("--images_folder", type=str, help="Path to folder of images that will be processed")
    parser.add_argument("--scribbles_folder", type=str, help="Path to folder of scribbles that will be used with images")
    
    # saving results for use cases intermediate and evaluation use cases (1 and 2)
    parser.add_argument("--output", type=str, help="Path to folder where all output will be saved")

    # for pipeline specification
    parser.add_argument("--max_img_dim", type=int, default=800, 
                        help="Number of pixels that the images maximum dimension is scaled to for processing")
    parser.add_argument("--feedback_thresh", type=int, default=0.01, 
                         help="Min proportional change in trimaps def area to pass back into refinement stage")

    # for masking stage specification
    parser.add_argument("--mask_config", type=str, default="Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
                        help="YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)")
    parser.add_argument("--instance_thresh", type=float, default=0.05, 
                        help="Mask R-CNN score threshold for instance recognition")

    # for trimap stage specification
    parser.add_argument("--def_fg_thresh", type=float, default=0.99,
                        help="Threshold above which mask pixels labelled as def fg for trimap network training")
    parser.add_argument("--unknown_thresh", type=float, default=0.1,
                        help="Threshold below which mask pixels labelled as def bg for trimap network training")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate during training of trimap network")
    parser.add_argument("--batch_size", type=int, default=12000, 
                        help="Batch size during training trimap network")    
    parser.add_argument("--unknown_lower_bound", type=float, default=0.3,
                        help="Probability below which trimap network inference is classified as bg")
    parser.add_argument("--unknown_upper_bound", type=float, default=0.7,
                        help="Probability above which trimap network inference is classified as fg")
    parser.add_argument("--no_optimisation", action='store_false', 
                        help="Do not perform trimap network training/inference process")

    # for refinement stage specification
    parser.add_argument("--matting_weights", type=str, default="./matting_network/FBA.pth", 
                        help="Path to pre-trained matting model")

    args = parser.parse_args()

    if args.intermediate and (args.image is None or args.output is None):
        parser.error("-intermediate requires --image and --output arguments!")


    if args.eval and (args.images_folder is None or args.output is None):
        parser.error("-eval requires --images_folder and --output arguments!")


    return args


def process_images_eval(args, pipeline):
    img_files = sorted(os.listdir(args.images_folder))
        
    if args.scribbles is not None:
        scribbles_files = sorted(os.listdir(args.scribbles_folder))
        assert img_files == scribbles_files, "Image and scribble folders must be identical!" 
        scribbles_iter = iter(sribble_files)
    
    for img_file in img_files:
        img = cv2.imread(os.path.join(args.images_folder, img_file))
        
        scribbles =  None
        if args.scribbles is not None: 
            scribbles = cv2.imread(os.path.join(args.scribbles, next(scribbles_iter)), 0).astype(np.int32)
            scribbles = preprocess_scribbles(scribbles_path)

        results = pipeline(img, scribbles)
        
        save_eval_output(results, img_file, args.output)


def save_eval_output(results, img_file, output_dir):
    trimaps_folder, fgs_folder, alphas_folder, mattes_folder = setup_eval_output(output_dir)

    cv2.imwrite(os.path.join(trimaps_folder, img_file), results["trimaps"][-1])
    cv2.imwrite(os.path.join(fgs_folder, img_file), results["foregrounds"][-1])
    cv2.imwrite(os.path.join(alphas_folder, img_file), results["alphas"][-1])
    cv2.imwrite(os.path.join(mattes_folder, img_file), results["mattes"][-1])


def setup_eval_output(parent_dir):
    trimaps_folder = os.path.join(parent_dir, "trimap")
    fgs_folder =  os.path.join(parent_dir, "fg")
    alphas_folder = os.path.join(parent_dir, "alpha")
    mattes_folder = os.path.join(parent_dir, "matte")   
    
    make_dirs(trimaps_folder, fgs_folder, alphas_folder, mattes_folder)

    return trimaps_folder, fgs_folder, alphas_folder, mattes_folder


def process_image_intermediate(args, pipeline):
    img = cv2.imread(args.image)
    
    scribbles = None
    if args.scribbles is not None:
        scribbles = cv2.imread(args.scribbles, 0).astype(np.int32)
        scribbles = preprocess_scribbles(scribbles, img)
       
    results = pipeline(img, scribbles)

    img_id = os.path.splitext(os.path.basename(args.image))[0]   
    save_intermediate_results(results, img_id, args.output)


def save_intermediate_results(results, img_id, output_folder):
    make_dirs(output_folder)
    for i, (img_type, pred) in enumerate(results.items()):
        save_results_type(pred, img_id, "{}_{}".format(i, img_type), output_folder)


def save_results_type(pred, img_id, output_type, output_folder):
    if type(pred) == list: 
        for i, img in enumerate(pred): 
            output_file_name = "{0}_{1}_iter{2}{3}".format(img_id, output_type, i, ".png")
            cv2.imwrite(os.path.join(output_folder, output_file_name), img)
    else: 
        output_file_name = "{0}_{1}{2}".format(img_id, output_type, ".png")
        cv2.imwrite(os.path.join(output_folder, output_file_name), pred)


def preprocess_scribbles(scribbles, img):
    assert img.shape[:2] == scribbles.shape[:2], (
        "Image: {} and Scribbles: {} must be same shape!".format(img.shape[:2], scribbles.shape[:2]))

    scribbles[scribbles == 128] = -1 # convert unnannotated pixels
    scribbles[scribbles == 255] = 1 # convert fg scribbles
    scribbles[scribbles == 0] = 0 # convert bg scribbles

    return scribbles


def setup_logging(args):
    pipe_logger = setup_logger("pipeline", "logs/pipeline.log")
    
    if args.no_logs:
        pipe_logger.disabled = True
    elif not os.path.exists("logs"):
        os.mkdir("logs")

    return pipe_logger


def make_dirs(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)

if __name__ == "__main__":
    main()
