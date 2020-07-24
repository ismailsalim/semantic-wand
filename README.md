# Semantic Wand: A Learning-based Tool for Object Extraction in Natural Images

## Setting up an environment
The pipeline is built using Python 3.6 and requires:
- PyTorch 1.4.0+ (Author uses torch==1.5.0+cu101 and torchvision==0.6.0+cu101)
- Detectron2 (Author uses detectron2==0.1.3+cu101)
- pycocotools
- OpenCV

You can install the correct version of Detectron2 (with cuda) according to the instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

**Note:** The masking stage relies on a pre-trained Mask R-CNN model provided by [Detectron2](https://github.com/facebookresearch/detectron2). The model should be downloaded automatically the first time it is used for inference so be aware that this first time will take a bit longer!

**Note:** The refinement stage relies on the pre-trained [FBA matting](https://github.com/MarcoForte/FBA_Matting) model, which **must** be manually downloaded from [here](https://drive.google.com/file/d/1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1/view) and placed in the `matting_network/` directory (see [Directory Structure](##directory-structure) below). This model is covered by the [Adobe Deep Image Matting Dataset License Agreement](https://drive.google.com/file/d/1MKRen-TDGXYxm9IawPAZrdXQIYhI0XRf/view).

## Use
### 1. Interactive app
The app is built with Python's Tk interface. Hopefully, usage is self-explanatory from the interface.

To start the app, run from the command line:
```bash
python main.py --interactive
```

### 2. Script for complete intermediate pipeline results
This script reads one image (.jpg or .png) specified in the command line and saves all the intermediate predictions and final matte in directories that can also be specified (directories must exist already).

Matting will only be done for the **largest identified object in the image**.

### Example usage of arguments
```bash
# This finds donkey.png in ./examples/donkey/ and feeds the image into the 
# pipeline using the default Mask R-CNN pre-trained model.
python main.py --input_img donkey.png

# This changes the maximum dimension of the image fed into the pipeline to 3000 
pixels (preserving the aspect ratio). 
# To maintain the original input image size, make this value bigger than the 
# largest dimension of the input image.
python main.py input_img donkey.png --max_img_dim 3000

# This changes the learning rate and batch size used for the trimap generation
# network to 0.0001 and 4000 respectively and will launch the interactive demo
python main.py --interactive --lr 0.0001 --batch_size 4000

# This uses a non-default model from Detectron2. Format follows directory structure 
# in the Detectron2 source code: 
# https://github.com/facebookresearch/detectron2/tree/master/configs
python main.py --input_img donkey.png --mask_config COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

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
├── examples/
│   ├── img_id_1 # directory with input image (same id as the image file)
│   │   └── img_id_1.ext # .png or .jpg (same id as the image directory)
│   └── img_id_2
│       └── img_id_2.ext
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

**Note:** The only thing that you have to do is add `FBA.pth` into `matting_network/` and esnure images used for `main.py` follow the specified structure show in `examples/` (for easiest use).





