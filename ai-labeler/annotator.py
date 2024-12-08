import cv2
import numpy as np
import torch
from torchvision import models, transforms

class AIAnnotator:
    def __init__(self):
        # Load pre-trained object detection model (Faster R-CNN in this case)
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # Set model to evaluation mode

        # Define the transform to apply to input image
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to tensor
        ])

    def annotate(self, image):
        # Convert the OpenCV image (BGR) to RGB for PyTorch model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the necessary transformations
        image_tensor = self.transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Perform object detection
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Extract bounding boxes and labels
        boxes = prediction[0]['boxes'].cpu().numpy()  # Bounding boxes
        labels = prediction[0]['labels'].cpu().numpy()  # Labels of detected objects

        # Filter out low-confidence predictions (optional)
        score_threshold = 0.5
        filtered_boxes = []
        for i in range(len(boxes)):
            if prediction[0]['scores'][i] > score_threshold:
                filtered_boxes.append(boxes[i])

        return filtered_boxes
