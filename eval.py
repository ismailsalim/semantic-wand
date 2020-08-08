from utils.logger import setup_logger
from utils.errors import *

import argparse
import os
from collections import defaultdict

import cv2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--preds', type=str, required=True,
                        help='Path to alpha or foreground predictions')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Path to ground truth alphas or foregrounds')
    parser.add_argument('--fg_weights', type=str,
                        help='Path to foreground prediction weights (i.e. ground truth alphas)')
    parser.add_argument('--ref', 
                        help='For reference to specific model for logging')
    args = parser.parse_args()
   
    if not os.path.exists("logs"):
        os.mkdir("logs")   
    eval_logger = setup_logger('evaluation', 'logs/eval.log')

    # Assumes file ids in both folders will be sorted identically for matching 
    # predictions with valid ground truth!
    pred_files = sorted(os.listdir(args.preds))
    target_files = sorted(os.listdir(args.target))
    assert len(pred_files) == len(target_files), (
        "Preds and ground truth folders must be the same size!")

    eval_logger.info("\n\n STARTING EVALUATION \n")

    if args.ref is not None:
        eval_logger.info("Model reference: {} \n".format(args.ref))
    
    errors = defaultdict(list)
    
    if args.fg_weights is not None: # evaluating foreground predictions
        weights_files = sorted(os.listdir(args.fg_weights))
        assert len(pred_files) == len(weights_files), (
             "Preds and fg_weights folders must be the same size!")
        
        for pred_id, target_id, weights_id in zip(pred_files, target_files, weights_files):
            pred, target, fg_weights = preprocess_images(os.path.join(args.preds, pred_id),
                                                        os.path.join(args.target, target_id),
                                                        1, # cv2 BGR image reading
                                                        os.path.join(args.fg_weights, weights_id)) 

            mse = compute_mse_error(pred, target, fg_weights) 
            sad = compute_sad_error(pred, target, fg_weights)
            errors['mse'].append(mse)
            errors['sad'].append(sad)

            eval_logger.info("Prediction file id: {}".format(pred_id))
            eval_logger.info("Ground truth file id: {}".format(target_id))
            eval_logger.info("MSE: {0:.4f}, SAD: {1:.4f}\n".format(mse, sad))

        report_summary(eval_logger, errors)


    else: # evaluating alpha predictions
        for pred_id, target_id in zip(pred_files, target_files):
            pred, target = preprocess_images(os.path.join(args.preds, pred_id),
                                            os.path.join(args.target, target_id),
                                            0) # cv2 GRAY image reading 

            gradient = compute_gradient_error(pred, target) 
            connectivity = compute_connectivity_error(pred, target, 0.1)
            mse = compute_mse_error(pred, target) 
            sad = compute_sad_error(pred, target)

            errors['grad'].append(gradient)
            errors['connectivity'].append(connectivity)
            errors['mse'].append(mse)
            errors['sad'].append(sad)
            
            eval_logger.info("Prediction file id: {}".format(pred_id))
            eval_logger.info("Ground truth file id: {}".format(target_id))
            eval_logger.info("Gradient: {0:.4f}, Connectivity: {1:.4f}, MSE: {2:.4f}, SAD: {3:.4f}\n".format(
            gradient, connectivity, mse, sad))
            
        report_summary(eval_logger, errors, for_alpha_preds=True)


def preprocess_images(pred_file, target_file, read_type, weights_file=None): 
    pred = cv2.imread(pred_file, read_type) 
    h, w = pred.shape[:2]
    target = cv2.imread(target_file, read_type)
    target = cv2.resize(target, (w, h), cv2.INTER_NEAREST) 

    if weights_file is not None:
        fg_weights = cv2.imread(weights_file, 0)
        fg_weights = cv2.resize(fg_weights, (w, h), 0, cv2.INTER_NEAREST) 
        return pred, target, fg_weights
    
    return pred, target


def report_summary(eval_logger, errors, for_alpha_preds=False):
    eval_logger.info("********************************************************")
    eval_logger.info("Average MSE: {0:.4f}".format(average(errors['mse'])))
    eval_logger.info("Average SAD: {0:.4f}".format(average(errors['sad'])))

    if for_alpha_preds:
        eval_logger.info("Average Gradient: {0:.4f}".format(average(errors['grad'])))
        eval_logger.info("Average Connectivity: {0:.4f}".format(average(errors['connectivity'])))

    eval_logger.info("********************************************************")


if __name__ == '__main__':
    main()