from utils.logger import setup_logger
from utils.errors import *

import argparse
import os
from collections import defaultdict

import cv2

def main():
    args = parse_args()
    if not os.path.exists("logs"):
        os.mkdir("logs")   
    eval_logger = setup_logger('evaluation', 'logs/eval.log')

    # Assumes file ids in both folders will be sorted identically for matching 
    # predictions with valid ground truth!
    pred_files = sorted(os.listdir(args.preds_folder))
    target_files = sorted(os.listdir(args.targets_folder))
    assert len(pred_files) == len(target_files), (
        "Preds and ground truth folders must be the same size!")

    if args.fg_weights_folder is not None: 
        weights_files = sorted(os.listdir(args.fg_weights_folder))
        assert len(pred_files) == len(weights_files), (
             "Preds and weights folders must be the same size!")
        weights_iter = iter(weights_files)

    eval_logger.info("\n\n STARTING EVALUATION \n")
    errors = defaultdict(list)
    
    for pred_file, target_file in zip(pred_files, target_files):    
        weights = None
        
        if args.fg_weights_folder is not None: # evaluating foreground predictions
            pred, target, weights = preprocess_images(os.path.join(args.preds_folder, pred_file),
                                                        os.path.join(args.targets_folder, target_file),
                                                        1, # cv2 bgr image reading for foreground
                                                        os.path.join(args.fg_weights_folder, next(weights_iter)))        
        
        else: # evaluating alpha predictions
            pred, target = preprocess_images(os.path.join(args.preds_folder, pred_file),
                                            os.path.join(args.targets_folder, target_file),
                                            0) # cv2 grayscale image reading for alpha
            gradient = compute_gradient_error(pred, target) 
            connectivity = compute_connectivity_error(pred, target, 0.1)
            errors['grad'].append(gradient)
            errors['connectivity'].append(connectivity)
        
        mse = compute_mse_error(pred, target, weights) 
        sad = compute_sad_error(pred, target, weights)
        errors['mse'].append(mse)
        errors['sad'].append(sad)

        eval_logger.info("File id: {}".format(pred_file))
        if args.fg_weights_folder:
            eval_logger.info("MSE: {0:.4f}, SAD: {1:.4f}\n".format(mse, sad))
        else:
            eval_logger.info("Gradient: {0:.4f}, Connectivity: {1:.4f}, MSE: {2:.4f}, SAD: {3:.4f}\n".format(
            gradient, connectivity, mse, sad))

    report_summary(eval_logger, errors, args.fg_weights_folder is not None)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--preds_folder', type=str, required=True,
                        help='Path to alpha or foreground predictions')
    parser.add_argument('-t', '--targets_folder', type=str, required=True,
                        help='Path to ground truth alphas or foregrounds')
    parser.add_argument('-w', '--fg_weights_folder', type=str,
                        help='Path to ground truth alphas for foreground evaluation')
    args = parser.parse_args()
    
    return args


def preprocess_images(pred_file, target_file, read_type, weights_file=None): 
    pred = cv2.imread(pred_file, read_type) 
    h, w = pred.shape[:2]
    target = cv2.imread(target_file, read_type)
    target = cv2.resize(target, (w, h), cv2.INTER_NEAREST) 

    if weights_file is not None:
        weights = cv2.imread(weights_file, 0)
        weights = cv2.resize(weights, (w, h), 0, cv2.INTER_NEAREST) 
        return pred, target, weights
    
    return pred, target


def report_summary(eval_logger, errors, for_fg_preds):
    eval_logger.info("********************************************************")
    eval_logger.info("Average MSE: {0:.4f}".format(average(errors['mse'])))
    eval_logger.info("Average SAD: {0:.4f}".format(average(errors['sad'])))

    if not for_fg_preds:
        eval_logger.info("Average Gradient: {0:.4f}".format(average(errors['grad'])))
        eval_logger.info("Average Connectivity: {0:.4f}".format(average(errors['connectivity'])))

    eval_logger.info("********************************************************")


if __name__ == '__main__':
    main()