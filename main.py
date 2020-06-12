from pipeline.pipe import Pipeline

import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # for image directory specification
    parser.add_argument('--images_dir', default='./examples/1-input_images', 
                        help='directory of input images to be processed')
    parser.add_argument('--instance_preds_dir', default='./examples/2-instance_preds', 
                        help='directory to save intermediate coarse instance segementation predictions')
    parser.add_argument('--subj_masks_dir', default='./examples/3-subj_mask_preds', 
                        help='directory to save intermediate coarse subject mask predictions')
    parser.add_argument('--trimaps_dir', default='./examples/4-trimaps', 
                        help='directory to save intermediate trimaps')
    parser.add_argument('--final_mattes_dir', default='./examples/5-final_mattes', 
                        help='directory to save final matting output')
    
    # for coarse stage specification
    parser.add_argument('--coarse_config', default='mask_rcnn_X_101_32x8d_FPN_3x.yaml', 
                        help='Detectron2 YAML file with Mask R-CNN configuration')
    parser.add_argument('--coarse_thresh', default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')

    # for trimap stage specification
    parser.add_argument('--trimap_kernel_size', default=5, 
                        help='Size of kernel used for trimap erosion/dilation')
    parser.add_argument('--dilation', default=3, 
                        help='Number of iterations applied for dilation')
    parser.add_argument('--erosion', default=0, 
                        help='Number of iterations applied for erosion')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting/FBA.pth', 
                        help='Size of kernel used for trimap erosion/dilation')

    args = parser.parse_args()

    pipeline = Pipeline(args)

    pipeline.process(args)

if __name__ == '__main__':
    main()
