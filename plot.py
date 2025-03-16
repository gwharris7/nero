import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import matplotlib.pyplot as plt

# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Create predictor
predictor = DefaultPredictor(cfg)

areas = np.zeros((3, 3))
scores = np.zeros((3, 3))
divergence = np.zeros((3, 3))

for i in range(9):
    path = f"chair/chair_crop_{i}.jpg" # Insert image path here

    image = cv2.imread(path)
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
    for k, pred_mask in enumerate(pred_masks):
        if pred_classes[k] == 56: # replace with class id of the object you want to detect
            mask = pred_mask.cpu().numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours.append(contour[0])
            break
    
    # draw each contour to the empty mask
    for contour in contours:
        cv2.drawContours(binary_mask, [contour], -1, (255,255,255), thickness=cv2.FILLED)

    row = i // 3
    col = i % 3

    for l, score in enumerate(pred_scores):
        if pred_classes[l] == 56:
            scores[row, col] = score
            break

    # store the areas in a list
    areas[row, col] = cv2.contourArea(contours[0]) if contours else None
    
    # Define shifts to move the window around
    shifts = [(width // 12, 0), (width // 12, 0), (0, -height // 12),
              (width // 12, 0), (width // 12, 0), (0, -height // 12),
              (width // 12, 0), (width // 12, 0), (0, -height // 12)]

    # Find the predicted location and calculate divergence
    for j, box in enumerate(pred_boxes):
        if i == 0: # use first box as expected location
            expected_x = (box[0] + box[2]) / 2
            expected_y = (box[1] + box[3]) / 2
        else:
            if pred_classes[j] == 56:  # Replace with class ID for chosen object
                if i == 3 or i == 6:
                    expected_x = 0
                expected_x = expected_x + shifts[i][0]
                expected_y = expected_y + shifts[i][1]
                pred_x = (box[0] + box[2]) / 2  # Predicted center x
                pred_y = (box[1] + box[3]) / 2  # Predicted center y
                divergence[i // 3, i % 3] = np.sqrt((pred_x - expected_x) ** 2 + (pred_y - expected_y) ** 2)
                if i == 8:
                    divergence[0, 0] = np.mean(divergence)
                break

    # find the pred_boxes index with corresponding type 8
    for j, box in enumerate(pred_boxes):
        if pred_classes[j] == 56:
            left = pred_boxes[j][0]
            bottom = pred_boxes[j][1]
            right = pred_boxes[j][2]
            top = pred_boxes[j][3]
            break
    
    print(f"Object {i}, type: {pred_classes[j]}, left = {left}, right = {right}, top = {top}, bottom = {bottom}")

    # Visualize results
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    output_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    # Save outputs
    path_obj = path.split("/")[0]
    output_path = f"{path_obj}/output_{path_obj}_crop_{i}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(output_image.get_image(), cv2.COLOR_RGB2BGR))
    print(f"Output saved to {output_path}")

# Create a 2D heatmap
fig, ax = plt.subplots()
cax = ax.imshow(divergence, cmap='viridis', interpolation='nearest')
fig.colorbar(cax)

ax.set_title('Divergence from Expected Center by Cropping Position')
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()
