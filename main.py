from pipeline.pipe import Pipeline

import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images_dir', default='./examples/input_images', 
                        help='directory of input images to be processed')
    parser.add_argument('--instance_preds_dir', default='./examples/instance_preds', 
                        help='directory to save intermediate coarse instance segementation predictions')
    parser.add_argument('--subj_masks_dir', default='./examples/subj_mask_preds', 
                        help='directory to save intermediate coarse subject mask predictions')

    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.process(args)

if __name__ == '__main__':
    main()
