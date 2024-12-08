import os
import json

class AnnotationSaver:
    def save(self, annotations, output_path):
        pass

class YOLOAnnotationSaver(AnnotationSaver):
    def save(self, annotations, output_path):
        with open(output_path, 'w') as file:
            for annotation in annotations:
                # Here annotation is a tuple (x1, y1, x2, y2)
                x1, y1, x2, y2 = annotation
                width, height = x2 - x1, y2 - y1
                x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                # Assuming label as '0' (or modify as needed)
                label = 0
                file.write(f"{label} {x_center} {y_center} {width} {height}\n")

class PascalVOCAnnotationSaver(AnnotationSaver):
    def save(self, annotations, output_path):
        xml_content = '<annotation>\n'
        for annotation in annotations:
            # Here annotation is a tuple (x1, y1, x2, y2)
            x1, y1, x2, y2 = annotation
            xml_content += f"""
            <object>
                <name>object</name>
                <bndbox>
                    <xmin>{x1}</xmin>
                    <ymin>{y1}</ymin>
                    <xmax>{x2}</xmax>
                    <ymax>{y2}</ymax>
                </bndbox>
            </object>\n"""
        xml_content += '</annotation>'
        
        with open(output_path, 'w') as file:
            file.write(xml_content)

class COCOAnnotationSaver(AnnotationSaver):
    def save(self, annotations, output_path):
        coco_annotations = []
        for annotation in annotations:
            # Here annotation is a tuple (x1, y1, x2, y2)
            x1, y1, x2, y2 = annotation
            coco_annotations.append({
                "category_id": 1,  # Assuming category_id as 1 (modify as needed)
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # Convert to float
                "score": 1.0  # Assuming score as 1.0 (modify as needed)
            })

        with open(output_path, 'w') as file:
            json.dump(coco_annotations, file)
