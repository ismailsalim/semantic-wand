![](images/demo.gif)

# Semantic Wand: A Learning-based Tool for Object Extraction in Natural Images

## Setting up an environment
The pipeline is implemented using Python 3.6 and requires:
- PyTorch (tested with 1.5.0+cu101 and torchvision==0.6.0+cu101)
- Detectron2 (tested with detectron2==0.1.3+cu101)
- pycocotools
- OpenCV

You can install the correct version of Detectron2 (with cuda) according to the instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

**Note:** The masking stage relies on a pre-trained Mask R-CNN model provided by [Detectron2](https://github.com/facebookresearch/detectron2). The model should be downloaded automatically the first time it is used for inference so be aware that this first time will take a bit longer!

**Note:** The refinement stage relies on the pre-trained [FBA matting](https://github.com/MarcoForte/FBA_Matting) model, which **must** be manually downloaded from [here](https://drive.google.com/file/d/1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1/view) and placed in the `matting_network/` directory (see [Directory Structure](##directory-structure) below). **This model is covered by the [Adobe Deep Image Matting Dataset License Agreement](https://drive.google.com/file/d/1MKRen-TDGXYxm9IawPAZrdXQIYhI0XRf/view)**.

**DISCLAIMER:** We acknowledge that this tool could be used for unethical image manipulation. We condemn the use of this tool for any such purpose and trust that users understand what falls under this category, preventing themselves from carrying out such activity. 

## Use
### 1. Interactive app
The app is built with Python's Tk interface. Hopefully, usage is self-explanatory.

To start the app, run from the command line:
```bash
python main.py -interactive
```

### 2. Complete intermediate pipeline results
```bash
# This runs the specified image with the specified scribble and saves all the 
# intermediate and final results that generated throughout the pipeline such as 
# the various trimap iterations.
python main.py -intermediate --image img.png --scribbles scribble.png --output output/
```

### 3. Multiple image processing (for evaluation)
```bash
# This runs every image in the specified input directory and their corresponding 
# scribble through the pipeline and outputs fg, alpha, and matte predictions to 
# the specified output folders. 
# - Assumes that scribbles and images are named identically!
# - Useful to produce the output to run `eval.py` for accuracy statistics.
python main.py -eval --images_folder imgs/ --scribles_folder scribbles/ --output output/
```

### Optional arguments
```bash
# This describe all the various arguments you can change
python main.py -h

# This changes the maximum dimension of the image fed into the pipeline to 1000 
# pixels large.
# To maintain the original input image size, make this value bigger than the largest 
# dimension of the input image.
python main.py  ... --max_img_dim 1000

# This changes the threshold applied for proceeding to the next iteration of the 
# refinement loop i.e the minimum proportional change in the trimap (a smaller 
# threshold tends towards more iterations).
python main.py ... --threshold 0.001

# This uses a non-default model from Detectron2. Format follows directory structure 
# in the Detectron2 source code: 
# https://github.com/facebookresearch/detectron2/tree/master/configs
python main.py ... --mask_config COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

# This changes the thresholds applied to the probability mask (heatmap) to generate 
# the definite foreground/background regions of the trimap
python main.py ... --def_fg_thresh 0.2 --def_bg_thresh 0.8

# This simply uses probability mask thresholding to generate the trimap and does 
# not apply the trimap generator network (faster but generally less accurate)
python main.py ... --no_optimisation

# This changes the thresholds applied to the sigmoid output to classify pixels 
# according to the trimap's three classes
python main.py ... --unknown_lower_bound 0.4 --unknown_upper_bound 0.6
```

## Evaluation
`eval.py` can be used to calculate alpha prediction errors with respect to ground truth alpha 

- Background images used by the author can be downloaded from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

- Ground truth foreground and alphas used by the author were selected from Adobe Composition 1k Dataset, which can be requested from the [Deep Image Matting](https://sites.google.com/view/deepimagematting) authors.

- Scripts for compositing foreground and background with the alpha can be found in `utils/`


## Directory Structure
For easiest use, keep to the following structure:
```bash
.
├── demo/
│   ├── app.py
│   ├── canvas.py
│   └── controller.py
├── masking_network/
│   ├── models.py
│   └── predictor.py
├── matting_network/
│   ├── FBA.pth  # PLACE PRE-TRAINED MATTING MODEL HERE (DOWNLOAD URL ABOVE)
│   ├── layers_WS.py
│   ├── models.py
│   └── resnet_GN_WS.py
├── pipeline/
│   ├── masking_stage.py
│   ├── pipe.py
│   ├── refinement_stage.py
│   └── trimap_stage.py
├── trimap_network/ 
│   └── models.py
├── utils/ 
├── eval.py
└── main.py
```





