# Semantic Wand: A Tool For Natural Image Object Extraction

## Setting up an environment
The tool is built using Python 3.6 and requires:
- PyTorch 1.4.0+ (I'm using torch==1.5.0+cu101 and torchvision==0.6.0+cu101)
- Detectron2 (I'm using detectron2==0.1.3+cu101)
- pycocotools
- OpenCV

You can install Detectron2 according to the instructions [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Use
Currently, the tool reads one image (.jpg or .png) specified in the command line and saves all the intermediate predictions and final matte (.png) in directories that can also be specified (directories must exist already). 

The coarse stage relies on pre-trained Mask R-CNN models provided by [Detectron2](https://github.com/facebookresearch/detectron2). A model is downloaded the first time it is specified in the command line when running the tool (see [Example usage](###Example-usage)). A reference to all the different pre-trained models avaiable can be found [here](https://github.com/facebookresearch/detectron2/tree/master/configs).

The refinement stage relies on the pre-trained [FBA matting](https://github.com/MarcoForte/FBA_Matting) model, which can be downloaded from [here](https://drive.google.com/file/d/1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1/view).

For easiest use, keep the repo in the following structure:
```bash
.
├── examples/
│   ├── 1_input_images # directory containing input images
│   ├── 2_instance_preds
│   ├── 3_subj_mask_preds
│   ├── 4_trimaps
│   ├── 5_alphas
│   ├── 6_foregrounds
│   └── 7_final_mattes
├── main.py
├── matting_network
│   ├── FBA.pth  # pre-trained matting model
│   ├── layers_WS.py
│   ├── models.py
│   └── resnet_GN_WS.py
├── pipeline
    ├── coarse_stage.py
    ├── pipe.py
    ├── refinement_stage.py
    └── trimap_stage.py
```

### Example usage:
```bash
# This finds donkey.png in ./examples/1_input_images/ using the default Mask R-CNN pre-trained model.
python3 main.py donkey.png

# This uses a non-default model from Detectron2
python3 main.py donkey.png --coarse_config=COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

# This changes the maximum dimension of the image fed into the pipeline to 3000 pixels (preserving the aspect ratio). To maintain the original input image size, make this value bigger than the largest dimension of the input image.
python3 main.py donkey.png --max_img_dim=3000

# This changes the scale factor applied for calculating the kernel size to 20000. A larger value leads to a smaller kernel size and less erosion/dilation during the trimap stage. (This will be scrapped soon but it's interested to play around with).
python3 main.py donkey.png --kernel_scale_factor=20000
```