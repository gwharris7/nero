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
from PIL import Image

# Helper function to crop the image
def crop_image_variants(image_path):
    image = Image.open(f"images/{image_path}")
    width, height = image.size
    
    # Sets cropping boundary at 5/6 of image height and width
    window_width = (5 * width) // 6
    window_height = (5 * height) // 6
    
    # Define shifts to move the window around
    shifts = [(0, 0), (width // 12, 0), (2 * width // 12, 0),
              (0, height // 12), (width // 12, height // 12), (2 * width // 12, height // 12),
              (0, 2 * height // 12), (width // 12, 2 * height // 12), (2 * width // 12, 2 * height // 12)]
    
    filenames = []
    for i, (dx, dy) in enumerate(shifts):
        left = min(width - window_width, dx)
        top = min(height - window_height, dy)
        right = left + window_width
        bottom = top + window_height
        
        cropped_img = image.crop((left, top, right, bottom))
        path_obj = image_path.split(".")[0]
        filename = f"{path_obj}/{path_obj}_crop_{i}.jpg"
        cropped_img.save(filename)
        filenames.append(filename)
    return filenames

# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Create predictor
predictor = DefaultPredictor(cfg)

# Load the images
image_paths = ["water_bottle.jpg",
               "frisbee.jpg",
               "kite.jpg",
               "chair.jpg",
               "bottle.jpg",
               "soccer_ball.jpg",
               "cup.jpg",
               "dog.jpg",
               "boat.jpg",
               "traffic_light.jpg"]

for path in image_paths:
    all_paths = []
    all_paths.append(crop_image_variants(path))

for path in all_paths:

    for img in path:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Perform inference
        outputs = predictor(image)
        pred_boxes = outputs["instances"].pred_boxes.tensor
        pred_classes = outputs["instances"].pred_classes
        pred_scores = outputs["instances"].scores
        pred_masks = outputs["instances"].pred_masks

        # Extract the contour of each predicted mask and save its area
        height, width, channels = image.shape
        binary_mask = np.zeros((height, width, channels), dtype=np.uint8)
        contours = []
        for pred_mask in pred_masks:
            mask = pred_mask.cpu().numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours.append(contour[0])
        
        # draw each contour to the empty mask
        for contour in contours:
            cv2.drawContours(binary_mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)

        # store the areas in a list
        areas = [cv2.contourArea(contour) for contour in contours]

        for i, box in enumerate(pred_boxes):
            # order: left edge x value, bottom edge y val, right edge x val, top edge y val
            left = pred_boxes[i][0]
            bottom = pred_boxes[i][1]
            right = pred_boxes[i][2]
            top = pred_boxes[i][3]
            print(f"Object {i}, type: {pred_classes[i]}, left = {left}, right = {right}, top = {top}, bottom = {bottom}")

            

        # Visualize results
        v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
        output_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save outputs
        path_obj = img.split(".")[0]
        output_path = f"{path_obj}/output_{path_obj}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(output_image.get_image(), cv2.COLOR_RGB2BGR))
        print(f"Output saved to {output_path}")
