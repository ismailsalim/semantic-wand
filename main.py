from pipeline.pipe import Pipeline

import argparse
import os
import time
# import logging

import cv2

def main():
    # logging.basicConfig(filename='pipeline.log', level=logging.DEBUG, 
    #                 format='%(asctime)s:%(levelname)s:%(message)s')
    # logging.getLogger('matplotlib.font_manager').disabled = True

    # (refactor) change to YAML?
    parser = argparse.ArgumentParser()
    
    # for image specification
    parser.add_argument('img_file',
                        help='the name of a specific image to be processed (inc. extension)')
    parser.add_argument('--annotated_file', 
                    help='the name of a specific annotated image (inc. extension)')
    parser.add_argument('--img_dir', 
                        help='where input image(s) and results are stored (if not in .examples/img_filename/)')
    parser.add_argument('--max_img_dim', type=int, default=1500,
                        help='Number of pixels that the image\'s maximum dimension is scaled to for processing')

    # for masking stage specification
    parser.add_argument('--mask_config', default='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        help='YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)')
    parser.add_argument('--score_thresh', type=float, default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')
    parser.add_argument('--mask_thresh', type=float, default=0.5, 
                        help='Mask R-CNN probability threshold for conversion of soft mask to binary mask')

    # for trimap stage specification
    parser.add_argument('--def_fg_threshs', type=float, nargs='+', default=[0.97, 0.98, 0.99],
                        help='Mask R-CNN pixel probability thresholds used for definite foreground')
    parser.add_argument('--unknown_threshs', type=float, nargs='+', default=[0.2, 0.15, 0.1],
                        help='Mask R-CNN pixel probability threshold used for unknown region')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Size of kernel used for trimap erosion/dilation')
    parser.add_argument('--iterations', type=int, default=0,
                        help='Number of iterations for alpha/trimap feedback loop')

    args = parser.parse_args()

    pipeline = Pipeline(max_img_dim = args.max_img_dim,
                        mask_config = args.mask_config,
                        roi_score_threshold = args.score_thresh,
                        mask_threshold = args.mask_thresh,
                        def_fg_thresholds = args.def_fg_threshs,
                        unknown_thresholds = args.unknown_threshs,
                        matting_weights = args.matting_weights,
                        iterations = args.iterations) 

    img, annotated_img, img_id, output_dir = setup_io(args.img_file, 
                                                        args.annotated_file, 
                                                        args.img_dir)

    results = pipeline(img, annotated_img)

    save_results(results, img_id, output_dir)


def setup_io(img_file, annotated_file, img_dir):
    img_id = os.path.splitext(img_file)[0]  
    if img_dir is None:
        img_dir = os.path.join('./examples', img_id)

    img = cv2.imread(os.path.join(img_dir, img_file))
    if img is None:
        raise ValueError("Input image not found!")
    
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
