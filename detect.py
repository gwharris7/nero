from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' stands for nano, use 's', 'm', 'l', or 'x' for bigger models

# Run detection on an image
results = model("snowboard.jpg")

# Show results
print(results)