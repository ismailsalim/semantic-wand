# This script composes selected foreground and background images using the corresponding 
# alpha mattes.
# Original source: https://github.com/Yaoyi-Li/GCA-Matting/blob/master/Composition_code.py

import os
import math
import time
import warnings

import cv2 as cv2
import numpy as np
from tqdm import tqdm

fg_path = './data/evaluation/input/fg'
alpha_path = './data/evaluation/input/alpha'
bg_path = './data/evaluation/input/bg'
out_path = './data/evaluation/input/merged'

fg_files = os.listdir(fg_path)
alpha_files = os.listdir(alpha_path)
bg_files = os.listdir(bg_path)

fg_files.sort()
alpha_files.sort()
bg_files.sort()

def compose(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp

bg_iter = iter(bg_files)

for im_name in fg_files:
    im = cv2.imread(fg_path + im_name)
    alpha = cv2.imread(alpha_path + im_name, 0)
    h, w = im.shape[:2]
    
    bg_name = next(bg_iter)
    bg = cv2.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    
    if ratio > 1:
        bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)

    out = compose(im, bg, alpha, w, h)
    filename = out_path+im_name.split('.')[0]+'_'+bg_name.split('.')[0]+'.png'
    cv2.imwrite(filename, out)




