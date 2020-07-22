# This script takes images from categories of the Indoor Scenes dataset
# and places them into a directory which can be used as the background images
# directory for compose_dataset.py

import os
import shutil

dataset_path = "./data/indoor_dataset/"
output_path = "./data/testing/bg/"
fg_path = "./data/testing/fg/"
num_fgs = len(os.listdir(fg_path)) # a distinct bg for each testing image

categories = ['auditorium', 'corridor', 'office', 'warehouse']
bg_paths = [os.path.join(dataset_path, c) for c in categories]

for i, p in enumerate(bg_paths):
    for j, f in enumerate(os.listdir(p)):
        if j < num_fgs:
            img_path = os.path.join(p, f)
            dest_path = os.path.join(output_path, str(i)+"_"+f)
            shutil.move(img_path, dest_path)
