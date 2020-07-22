from pipeline.pipe import Pipeline
from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

from demo.app import App
from utils.eval import *

from utils.logger import setup_logger

import argparse
import os
import time
import tkinter as tk
import cv2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--interactive', action='store_true',
                        help='To use interactive demo with scribbles for object selection')

    # for evaluation
    parser.add_argument('--eval', action='store_true',
                        help='To use evaluate matting accuracy on testing dataset')
    parser.add_argument('--composited', default="./data/testing/merged/",
                        help='Path to evaluation images')
    parser.add_argument('--alphas',  default="./data/testing/alpha/",
                        help='Path to ground truth alphas')
    parser.add_argument('--fgs', default="./data/testing/fg/",
                        help='Path to ground truth foregrounds')
    parser.add_argument('--trimaps', default="./data/testing/trimaps/",
                        help='Path to ground truth trimaps')

    # image specification for non interactive usage
    parser.add_argument('--input_img',
                        help='Name of a specific image to be processed (inc. extension)')
    parser.add_argument('--annotations', 
                        help='Name of a specific annotated image (inc. extension)')
    parser.add_argument('--img_dir', 
                        help='Where input image(s) and results are stored (if not in .examples/img_id/)')
    parser.add_argument('--max_img_dim', type=int, default=1000, # set to 1920 for evaluation if using same dataset
                        help='Number of pixels that the image\'s maximum dimension is scaled to for processing')

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

    # for matting stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Path to pre-trained matting model')

    # for refinement feedback loop
    parser.add_argument('--feedback_thresh', type=int, default=0.01, 
                         help='Min proportional change in trimap\'s def area to pass back into refinement stage')

    args = parser.parse_args()

    if args.eval and args.interactive:
        raise ValueError("Can't use --eval and --interactive at the same time!")
    
    pipe_logger = setup_logger("pipeline", "pipeline.log")

    masking_stage = MaskingStage(args.mask_config, args.instance_thresh)
    trimap_stage = TrimapStage(args.def_fg_thresh, args.unknown_thresh,
                                args.lr, args.batch_size,
                                args.unknown_lower_bound, args.unknown_upper_bound)  
    refinement_stage = RefinementStage(args.matting_weights)
    
    pipeline = Pipeline(masking_stage, trimap_stage, refinement_stage, 
                        args.feedback_thresh, args.max_img_dim) 

    if args.interactive:
        root = tk.Tk()
        app = App(root, pipeline)
        root.deiconify()
        app.mainloop()
    
    elif args.eval:
        if set(os.listdir("./data/testing/alpha/")) != set(os.listdir("./data/testing/fg/")):
            raise ValueError("Fg directory and alpha directory must contain the same image filenames!")
        evaluate(pipeline, args)  
    
    else:
        if args.input_img is None:
            raise ValueError("Must specify an input image!")
        img, annotated_img, img_id, output_dir = setup_io(args.input_img, 
                                                        args.annotations, 
                                                        args.img_dir)
        results = pipeline(img, annotated_img)
        save_results(results, img_id, output_dir)


def evaluate(pipeline, args):
    eval_logger = setup_logger('evaluation', 'eval.log')

    output_path = os.path.join("./eval_results", time.strftime("%d-%m--%H-%M-%S"))    
    os.mkdir(output_path)

    alpha_files = os.listdir(args.alphas)
    alpha_ids = [f.split('.')[0] for f in alpha_files] # remove extension

    for img_file in tqdm(os.listdir(args.composited)):
        eval_logger.info("\nNEW INPUT IMAGE FILE: {}".format(img_file))
        img = cv2.imread(os.path.join(args.composited, img_file))

        req_max_dim = max(img.shape[:2])
        if args.max_img_dim < req_max_dim:
            raise ValueError("--max_img_dim set too low for current image which has max dim of" 
                            + " {}".format(req_max_dim))    
       
        try:
            results = pipeline(img)
            alpha = results["alphas"][-1] # final iteration
            fg = results["foregrounds"][-1] 
            matte = results["mattes"][-1]

            img_id = img_file.split('.')[0]
            cv2.imwrite(os.path.join(output_path, img_id+"_alpha.png"), alpha)
            cv2.imwrite(os.path.join(output_path, img_id+"_fg.png"), fg)
            cv2.imwrite(os.path.join(output_path, img_id+"_matte.png"), matte)

            id_match = [i for i in alpha_ids if i in img_file] # check if alpha id in composited
            if len(id_match) < 1:
                raise ValueError("Composed images filenames do not contain alpha filename ids")
            elif len(id_match) > 1:
                raise ValueError("Two distinct alpha image files with the same filename")

            alpha_id = id_match[0]
            alpha_file = next(a for a in alpha_files if alpha_id in a)
            gt_alpha = cv2.imread(os.path.join(args.alphas, alpha_file), 0)

            gradient = compute_gradient_error(alpha, gt_alpha) 
            connectivity = compute_connectivity_error(alpha, gt_alpha, 0.1)
            mse = compute_mse_error(alpha, gt_alpha) 
            sad = compute_sad_error(alpha, gt_alpha) 
            
            eval_logger.info("Gradient: {}, Connectivity: {}, MSE: {}, SAD: {}".format(
                gradient, connectivity, mse, sad
            ))

        except ValueError as e:
            eval_logger.error(str(e))
        except:
            eval_logger.error("SOMETHING UNEXPECTED WENT WRONG!")

        break  


def setup_io(img_file, annotated_file, img_dir):
    img_id = os.path.splitext(img_file)[0]  
    
    if img_dir is None:
        img_dir = os.path.join('./examples', img_id)

    img = cv2.imread(os.path.join(img_dir, img_file))
    
    if annotated_file:
        annotated_img = cv2.imread(os.path.join(img_dir, annotated_file))
    else:
        annotated_img = None

    output_dir = os.path.join(img_dir, time.strftime("%d-%m--%H-%M-%S"))
    os.mkdir(output_dir)

    return img, annotated_img, img_id, output_dir
    

def save_results(results, img_id, to_dir):
    for i, (img_type, pred) in enumerate(results.items()):
        save(pred, img_id, '{}_{}'.format(i, img_type), to_dir)


def save(pred, img_id, output_type, to_dir):
    if type(pred) == list: # trimaps, foregrounds, alphas, mattes
        for i, img in enumerate(pred): 
            output_file_name = '{0}_{1}_iter{2}{3}'.format(img_id, output_type, i, '.png')
            cv2.imwrite(os.path.join(to_dir, output_file_name), img)
    else: # instances, unknown_mask, fg_mask
        output_file_name = '{0}_{1}{2}'.format(img_id, output_type, '.png')
        cv2.imwrite(os.path.join(to_dir, output_file_name), pred)


if __name__ == '__main__':
    main()
