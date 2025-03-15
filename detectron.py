# pip install 'git+https://github.com/facebookresearch/detectron2.git'


import cv2
import torch
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Create predictor
predictor = DefaultPredictor(cfg)

# Load the images
# image_paths = ["water_bottle.jpg",
#                "frisbee.jpg",
#                "kite.jpg",
#                "chair.jpg",
#                "bottle.jpg",
#                "soccer_ball.jpg"
#                "cup.jpg",
#                "dog.jpg",
#                "boat.jpg",
#                "traffic_light.jpg"]

image_paths = ["traffic_light.jpg"]

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # Perform inference
    outputs = predictor(image)
    pred_boxes = outputs["instances"]._fields["pred_boxes"].tensor
    pred_classes = outputs["instances"]._fields["pred_classes"]


    for i, box in enumerate(pred_boxes):
        # order: left edge x value, bottom edge y val, right edge x val, top edge y val
        left = pred_boxes[i][0]
        bottom = pred_boxes[i][1]
        right = pred_boxes[i][2]
        top = pred_boxes[i][3]
        print(f"Object {i}: left = {left}, right = {right}")

    # Visualize results
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    output_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save or display output
    path_obj = path.split(".")[0]
    output_path = f"output_{path_obj}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(output_image.get_image(), cv2.COLOR_RGB2BGR))
    print(f"Output saved to {output_path}")