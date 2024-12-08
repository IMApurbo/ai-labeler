# labeler.py
import os
from .annotator import AIAnnotator
from .annotation_saver import YOLOAnnotationSaver, PascalVOCAnnotationSaver, COCOAnnotationSaver

class AILabeler:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.annotator = AIAnnotator(model_name, device)
    
    def label_images(self, image_dir, save_dir, format='YOLO', threshold=0.5):
        saver = self._get_saver(format)

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(root, file)
                    annotations = self.annotator.annotate_image(image_path)

                    if annotations:
                        relative_path = os.path.relpath(image_path, image_dir)
                        save_path = os.path.join(save_dir, relative_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        saver.save(annotations, save_path.replace(file, f'.txt'))

    def _get_saver(self, format):
        if format == 'YOLO':
            return YOLOAnnotationSaver()
        elif format == 'PascalVOC':
            return PascalVOCAnnotationSaver()
        elif format == 'COCO':
            return COCOAnnotationSaver()
        else:
            raise ValueError(f"Unsupported format: {format}")
