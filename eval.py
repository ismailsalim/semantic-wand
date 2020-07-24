from utils.logger import setup_logger
from utils.errors import *

import argparse
import os
from collections import defaultdict

import cv2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('alpha_preds',
                        help='Path to alpha predictions')
    parser.add_argument('alpha_gts',
                        help='Path to ground truth alphas')
    parser.add_argument('--ref', 
                        help='For reference to specific model for logging')


    if not os.path.exists("logs"):
        os.mkdir("logs")
        
    eval_logger = setup_logger('evaluation', 'logs/eval.log')

    args = parser.parse_args()

    # assumes file ids in both folders will be sorted identically for matching estimate with 
    # valid ground truth!
    # Works best with consistent naming across folders as per use of pre-processing/evaluation 
    # scripts in `./utils/`
    alpha_pred_files = sorted(os.listdir(args.alpha_preds))
    alpha_gt_files = sorted(os.listdir(args.alpha_gts))
    assert len(alpha_pred_files) == len(alpha_gt_files), "Preds and ground truth folders must be same size!"

    eval_logger.info("\n\n EVALUATION \n")

    if args.ref is not None:
        eval_logger.info("Model reference: {} \n".format(args.ref))
    
    errors = defaultdict(list)
    for pred_id, alpha_id in zip(alpha_pred_files, alpha_gt_files):
        pred = cv2.imread(os.path.join(args.alpha_preds, pred_id), 0) 

        h, w = pred.shape[:2]
        alpha = cv2.imread(os.path.join(args.alpha_gts, alpha_id), 0)
        alpha = cv2.resize(alpha, (w, h)) 

        gradient = compute_gradient_error(pred, alpha) 
        connectivity = compute_connectivity_error(pred, alpha, 0.1)
        mse = compute_mse_error(pred, alpha) 
        sad = compute_sad_error(pred, alpha) 

        errors['grad'].append(gradient)
        errors['connectivity'].append(connectivity)
        errors['mse'].append(mse)
        errors['sad'].append(sad)
        
        eval_logger.info("Alpha prediction file id: {}".format(pred_id))
        eval_logger.info("Alpha ground truth file id: {}".format(alpha_id))

        eval_logger.info("Gradient: {0:.4f}, Connectivity: {1:.4f}, MSE: {2:.4f}, SAD: {3:.4f}".format(
            gradient, connectivity, mse, sad
        ))

    eval_logger.info("********************************************************")
    eval_logger.info("Number of samples: {}".format(len(alpha_pred_files)))
    eval_logger.info("Average Gradient: {0:.4f}".format(average(errors['grad'])))
    eval_logger.info("Average Connectivity: {0:.4f}".format(average(errors['connectivity'])))
    eval_logger.info("Average MSE: {0:.4f}".format(average(errors['mse'])))
    eval_logger.info("Average SAD: {0:.4f}".format(average(errors['sad'])))
    eval_logger.info("********************************************************")


if __name__ == '__main__':

    main()