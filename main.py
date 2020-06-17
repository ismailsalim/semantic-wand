# local application libraries
from pipeline.pipe import Pipeline

# system libraries
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # for image specification
    parser.add_argument('img_filename',
                        help='the name of a specific image to be processed')
    parser.add_argument('--max_img_dim', default=1500,
                        help='Number of pixels that the image\'s maximum dimension is scaled to')
    
    # for loading images and saving results
    parser.add_argument('--images_dir', default='./examples/1-input_images', 
                        help='directory of the input image')
    parser.add_argument('--instance_preds_dir', default='./examples/2-instance_preds', 
                        help='directory to save intermediate coarse instance segementation prediction')
    parser.add_argument('--subj_masks_dir', default='./examples/3-subj_mask_preds', 
                        help='directory to save intermediate coarse subject mask prediction')
    parser.add_argument('--trimaps_dir', default='./examples/4-trimaps', 
                        help='directory to save intermediate trimap')
    parser.add_argument('--alphas_dir', default='./examples/5-alphas', 
                        help='directory to save intermediate matting alpha prediction')
    parser.add_argument('--fgs_dir', default='./examples/6-foregrounds', 
                        help='directory to save intermediate matting foreground prediction')
    parser.add_argument('--final_mattes_dir', default='./examples/7-final_mattes', 
                        help='directory to save final matting output')

    # for coarse stage specification
    parser.add_argument('--coarse_config', default='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
                        help='YAML file with Mask R-CNN configuration (see Detectron2 Model Zoo)')
    parser.add_argument('--coarse_thresh', default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')

    # for trimap stage specification
    parser.add_argument('--kernel_scale_factor', default=10000, 
                        help='Number to divide box area by to obtain kernel size')
    parser.add_argument('--kernel_shape', default='MORPH_RECT', 
                        help='OpenCV kernel shape type for erosion/dilation')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Size of kernel used for trimap erosion/dilation')

    args = parser.parse_args()

    pipeline = Pipeline(args) 

    pipeline.process()

if __name__ == '__main__':
    main()
