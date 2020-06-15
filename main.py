# local application libraries
from pipeline.pipe import Pipeline

# system libraries
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # for image directory specification
    parser.add_argument('img_file',
                        help='the name of a specific image to be processed')
    parser.add_argument('--images_dir', default='./examples/1-input_images', 
                        help='directory of the input image')
    parser.add_argument('--instance_preds_dir', default='./examples/2-instance_preds', 
                        help='directory to save intermediate coarse instance segementation prediction')
    parser.add_argument('--subj_masks_dir', default='./examples/3-subj_mask_preds', 
                        help='directory to save intermediate coarse subject mask prediction')
    parser.add_argument('--trimaps_dir', default='./examples/4-trimaps', 
                        help='directory to save intermediate trimap')
    parser.add_argument('--final_mattes_dir', default='./examples/5-final_mattes', 
                        help='directory to save final matting output')

    parser.add_argument('--max_dim', default=600,
                        help='Maximum dimension in pixels after resizing input image')
    
    # for coarse stage specification
    parser.add_argument('--coarse_config', default='COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml', 
                        help='Detectron2 YAML file with Mask R-CNN configuration: https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py')
    parser.add_argument('--coarse_thresh', default=0.8, 
                        help='Mask R-CNN score threshold for instance recognition')

    # for trimap stage specification
    parser.add_argument('--kernel_scale_factor', default=30000, 
                        help='Number to divide box area by to obtain kernel size')
    parser.add_argument('--kernel_shape', default='MORPH_RECT', 
                        help='OpenCV kernel shape type for erosion/dilation')

    # for refinement stage specification
    parser.add_argument('--matting_weights', default='./matting_network/FBA.pth', 
                        help='Size of kernel used for trimap erosion/dilation')

    args = parser.parse_args()

    pipeline = Pipeline(args) 

    pipeline.process(args)

if __name__ == '__main__':
    main()
