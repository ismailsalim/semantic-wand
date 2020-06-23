import detectron2.data.transforms as T
import torch

from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data import MetadataCatalog

# from detectron2.data import (
#     MetadataCatalog,
#     build_detection_test_loader,
#     build_detection_train_loader,
# )
# from detectron2.evaluation import (
#     DatasetEvaluator,
#     inference_on_dataset,
#     print_csv_format,
#     verify_results,
# )

from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler, build_optimizer
# from detectron2.utils import comm
# from detectron2.utils.collect_env import collect_env_info
# from detectron2.utils.env import seed_all_rng
# from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
# from detectron2.utils.logger import setup_logger
# from detectron2.structures import Instances

class Predictor:
  """
  Used to actually initiate model inference by:
  1. Loading checkpoint from `cfg.MODEL.WEIGHTS`.
  2. Taking BGR image as the input and applyubg conversion defined by `cfg.INPUT.FORMAT`.
  3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.

  NEED TO CHANGE THIS
  4. Take one input image and produce a single output, instead of a batch.
  """

  def __init__(self, cfg, mask_thresh):
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

    self.mask_thresh = mask_thresh


  def __call__(self, original_image):
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

      predictions = self.model([inputs], self.mask_thresh)

      return predictions