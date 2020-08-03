# This script can be used to overlay scribbles made in the interactive demo with
# images so that they can be used by the closed form matting implementation provided 
# by https://github.com/MarcoForte/closed-form-matting
#
# This is useful for direct comparison between the methods.

import os
import cv2
import numpy as np

images_path = './data/evaluation/input/merged' # composited original images
scribbles_path = './data/evaluation/input/scribbles' # plain scribbles
overlayed_path = './data/evaluation/input/overlayed_scribbles' # image + scribbles

def overlay_scibbles(img, scribbles):
    mask = (scribbles == 128)
    img_annot = np.copy(scribbles)
    img_annot[mask] = img[mask]
    return img_annot


image_files =  sorted(os.listdir(images_path))
scribble_files = sorted(os.listdir(scribbles_path))

assert image_files == scribble_files, "Scribble and image folders must be identical!"

for file_id in image_files:      
    scribbles = cv2.imread(os.path.join(scribbles_path, file_id), 1)
    img = cv2.imread(os.path.join(images_path, file_id), 1)
    img_annot = overlay_scibbles(img, scribbles)

    # both used as input to closed form matting
    cv2.imwrite(os.path.join(images_path, file_id), img)
    cv2.imwrite(os.path.join(overlayed_path, file_id), img_annot)




