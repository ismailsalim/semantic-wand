# local application libraries
from pipeline.pipe import Pipeline

# system libraries
import argparse
import os

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
    parser.add_argument('--coarse_config', default='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        help='YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)')
    parser.add_argument('--coarse_thresh', type=float, default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')
    parser.add_argument('--unknown_thresh', type=float, default=0.01, 
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
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations for alpha/trimap feedback loop')

    args = parser.parse_args()

    pipeline = Pipeline(args) 

    pipeline.process()

if __name__ == '__main__':
    main()
