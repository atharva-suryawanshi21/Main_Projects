These are checkpoints of yolov8m model

# for web cam
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("./best.pt")

# Inference using webcam (source="0" is for default webcam)
model.predict(source=0, show=True)


# for individual image

from ultralytics import YOLO
import cv2

# Load model
model = YOLO("./best.pt")

# Inference=pic.j[]-pic.jpg
results = model("image.jpg", show=True)  # or model.predict(source="0") for webcam

 
# Visualize
results[0].show()  # opens with bounding boxes
results[0].save(filename="output.jpg")  # save manually