from pipeline.pipe import Pipeline

import argparse
import os
import time
import logging

import cv2

def main():
    parser = argparse.ArgumentParser()
    
    # for image specification
    parser.add_argument('img_filename',
                        help='the name of a specific image to be processed')
    parser.add_argument('--img_dir', 
                        help='where input image and results are stored (default finds dir with same name as filename in .examples/)')
    parser.add_argument('--max_img_dim', type=int, default=1500,
                        help='Number of pixels that the image\'s maximum dimension is scaled to for processing')

    # for masking stage specification
    parser.add_argument('--mask_config', default='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        help='YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)')
    parser.add_argument('--mask_thresh', type=float, default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')
    parser.add_argument('--unknown_thresh', type=float, default=0.3, 
                    help='Mask R-CNN pixel probability threshold used for unknown region')
    parser.add_argument('--def_fg_thresh', type=float, default=0.99, 
                    help='Mask R-CNN pixel probability threshold used for definite foreground')

    # for trimap stage specification
    parser.add_argument('--kernel_scale_factor', type=int, default=10000, 
                        help='Number to divide box area by to obtain kernel size')
    parser.add_argument('--kernel_shape', default='MORPH_RECT', 
                        help='OpenCV kernel shape type for erosion/dilation')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Size of kernel used for trimap erosion/dilation')
    parser.add_argument('--iterations', type=int, default=0,
                        help='Number of iterations for alpha/trimap feedback loop')

    args = parser.parse_args()

    pipeline = Pipeline(max_img_dim = args.max_img_dim,
                        mask_config = args.mask_config,
                        mask_thresh = args.mask_thresh,
                        trimap_thresholds = [args.unknown_thresh, args.def_fg_thresh],
                        kernel_scale_factor = args.kernel_scale_factor,
                        kernel_shape = args.kernel_shape,
                        matting_weights = args.matting_weights,
                        iterations = args.iterations) 

    img, img_id, output_dir = setup_io(args.img_filename, args.img_dir)

    results = pipeline(img)

    save_results(results, img_id, output_dir)

    a = 5


def setup_io(img_file, img_dir):
    img_id = os.path.splitext(img_file)[0]  
    if img_dir is None:
        img_dir = os.path.join('./examples', img_id)

    img = cv2.imread(os.path.join(img_dir, img_file))
    if img is None:
        raise ValueError("Image not found!")
    
    output_dir = os.path.join(img_dir, time.strftime("%d-%m--%H-%M-%S"))
    os.mkdir(output_dir)
    logging.debug('\nResults to be saved in: {}'.format(output_dir))

    return img, img_id, output_dir
    

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
