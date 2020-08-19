import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model

import logging
import torch

class Predictor:
  """
  Used to actually initiate model inference by:
  1. Loading checkpoint from `cfg.MODEL.WEIGHTS`.
  2. Taking BGR image as the input and applyubg conversion defined by `cfg.INPUT.FORMAT`.
  3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
  4. Take one input image and produce a single output, instead of a batch.

  Note that this simply adds mask threshold specification to Detectron2's DefaultPredictor 
  in order to binarise instances' soft masks with varying thresholds.

  Source code for DefaultPredictor can be found here: 
  https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultPredictor
  """

  def __init__(self, cfg):
    self.cfg = cfg.clone() 
    self.model = build_model(self.cfg)
    self.model.eval()
    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(self.model) 
    checkpointer.load(cfg.MODEL.WEIGHTS)

    self.transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    self.input_format = cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format


  def __call__(self, original_image, mask_threshold):
    """
    Args:
      original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
      predictions (dict):
          the output of the model for one image only.
    """
    with torch.no_grad():  
      if self.input_format == "RGB":
          original_image = original_image[:, :, ::-1]
      
      height, width = original_image.shape[:2]
      image = self.transform_gen.get_transform(original_image).apply_image(original_image)
      image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

      inputs = {"image": image, "height": height, "width": width}

      predictions = self.model([inputs], mask_threshold)

      return predictions