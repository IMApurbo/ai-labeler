# model_loader.py
import torch
from torchvision import models

def load_model(model_name='fasterrcnn_resnet50_fpn'):
    if model_name == 'fasterrcnn_resnet50_fpn':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'yolov5':
        # Assuming you have a YOLOv5 implementation available
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.eval()
    return model
