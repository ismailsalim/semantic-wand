from pipeline.pipe import Pipeline
from pipeline.masking_stage import MaskingStage
from pipeline.trimap_stage import TrimapStage
from pipeline.refinement_stage import RefinementStage

from demo.app import App

import argparse
import os
import time
import tkinter as tk
import cv2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--interactive', action='store_true',
                        help='To use interactive demo with scribbles for object selection')

    # image specification for non interactive usage
    parser.add_argument('--input_img',
                        help='Name of a specific image to be processed (inc. extension)')
    parser.add_argument('--annotations', 
                        help='Name of a specific annotated image (inc. extension)')
    parser.add_argument('--img_dir', 
                        help='Where input image(s) and results are stored (if not in .examples/img_id/)')
    parser.add_argument('--max_img_dim', type=int, default=1000,
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

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Path to pre-trained matting model')

    args = parser.parse_args()
    
    # initialise stages of pipeline
    masking_stage = MaskingStage(args.mask_config, args.instance_thresh)
    trimap_stage = TrimapStage(args.def_fg_thresh, args.unknown_thresh,
                                args.lr, args.batch_size,
                                args.unknown_lower_bound, args.unknown_upper_bound)  
    refinement_stage = RefinementStage(args.matting_weights)
    
    pipeline = Pipeline(masking_stage, trimap_stage, refinement_stage, 
                        args.max_img_dim) 

    if args.interactive:
        root = tk.Tk()
        app = App(root, pipeline)
        root.deiconify()
        app.mainloop()
    else:
        assert args.input_img is not None, "Must specify an input image!"
        img, annotated_img, img_id, output_dir = setup_io(args.input_img, 
                                                        args.annotations, 
                                                        args.img_dir)
        results = pipeline(img, annotated_img)
        save_results(results, img_id, output_dir)


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
