# This script takes images that can be used as the background for composition with
# ground truth foregrounds and alphas that can be done with `compose_dataset.py`
# The PASCAL VOC dataset was used for the published results.

import os
import shutil

bgs_path = "./data/VOCdevkit/VOC2012/JPEGImages"
fgs_path = "./data/evaluation/input/fg" # ground truth foregrounds
outputs_path = "./data/evaluation/inpit/bg"

category= 'airport_inside' # a folder name in the Indoor Scenes dataset

num_bg_per_fg = 1 # change for more background per foreground image for composition
num_fgs = len(os.listdir(fgs_path ))

it = iter(os.listdir(bgs_path))

for i in range(num_bg_per_fg):
    for j in range(num_fgs):
        bg_id = next(it)
        bg_file = os.path.join(bgs_path, bg_id)
        dest_path = os.path.join(outputs_path, '_' + bg_id)
        shutil.move(bg_file, dest_path)

