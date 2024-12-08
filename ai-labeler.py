import cv2
import numpy as np
import torch
from torchvision import models, transforms
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import os
import json

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Labeler Tool")
        self.root.geometry("1000x600")

        # Initialize AI Annotator
        self.annotator = AIAnnotator()

        # Setup Canvas for displaying image
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill="both", expand=True)

        # Variables for bounding box drawing
        self.rect_start = None
        self.rect_end = None
        self.rect_id = None
        self.bboxes = []
        self.undo_stack = []
        self.redo_stack = []

        # Buttons for loading images and saving annotations
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(side="left", padx=10)

        self.edit_button = tk.Button(self.root, text="Edit Annotations", command=self.edit_annotations, state="disabled")
        self.edit_button.pack(side="left", padx=10)

        self.save_button = tk.Button(self.root, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(side="right", padx=10)

        self.next_button = tk.Button(self.root, text="Next Image", command=self.load_next_image)
        self.next_button.pack(side="right", padx=10)

        # Undo/Redo buttons
        self.undo_button = tk.Button(self.root, text="Undo", command=self.undo, state="disabled")
        self.undo_button.pack(side="left", padx=10)

        self.redo_button = tk.Button(self.root, text="Redo", command=self.redo, state="disabled")
        self.redo_button.pack(side="left", padx=10)

        # Format dropdown
        self.format_label = tk.Label(self.root, text="Select Format:")
        self.format_label.pack(side="left", padx=10)
        
        self.format_combobox = ttk.Combobox(self.root, values=["YOLO", "COCO", "Pascal VOC"])
        self.format_combobox.set("YOLO")  # Default to YOLO
        self.format_combobox.pack(side="left", padx=10)

        # Load initial image
        self.image_path = None
        self.image = None
        self.image_tk = None
        self.annotated_image = None
        self.annotations = []

        # Label for loading message
        self.loading_label = tk.Label(self.root, text="Loading...", font=("Helvetica", 12), fg="blue")
        self.loading_label.pack_forget()

        self.save_directory = None  # Variable to store the save directory path

    def load_image(self):
        # Display loading message
        self.loading_label.pack(side="top", pady=20)

        # Ask user to select any file (no filetype filter)
        self.image_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        
        if not self.image_path:
            self.loading_label.pack_forget()  # Hide loading message
            return

        # Load image with OpenCV
        self.image = cv2.imread(self.image_path)
        self.annotated_image = self.image.copy()

        # Annotate the image using AI (e.g., Faster R-CNN, YOLO, etc.)
        self.bboxes = self.annotator.annotate(self.image)

        # Hide loading message and display image
        self.loading_label.pack_forget()
        self.display_image()

        # Enable the edit button after loading image
        self.edit_button.config(state="normal")

    def display_image(self):
        # Convert image from OpenCV (BGR) to PIL (RGB)
        self.image_rgb = cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(self.image_rgb)
        
        # Resize image to fit in the window if it's too large
        width, height = self.image_pil.size
        max_width = self.canvas.winfo_width()
        max_height = self.canvas.winfo_height()

        if width > max_width or height > max_height:
            aspect_ratio = width / height
            if width > max_width:
                width = max_width
                height = int(width / aspect_ratio)
            if height > max_height:
                height = max_height
                width = int(height * aspect_ratio)

            self.image_pil = self.image_pil.resize((width, height))

        self.image_tk = ImageTk.PhotoImage(self.image_pil)

        # Create canvas item for image
        self.canvas.delete("all")  # Clear previous items on the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

        # Draw bounding boxes from annotations
        for bbox in self.bboxes:
            self.draw_bbox(bbox)

    def draw_bbox(self, bbox):
        # Draw bounding box on the image
        x1, y1, x2, y2 = bbox
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def save_annotations(self):
        # Prompt for saving directory if not already set
        if not self.save_directory:
            self.save_directory = filedialog.askdirectory(title="Select Directory to Save Annotations")
        
        if not self.save_directory:
            self.save_directory = os.path.dirname(self.image_path)  # Save in the same directory as the image
        
        # Get selected format from the dropdown
        format_selected = self.format_combobox.get()

        # Prepare output file path
        file_name = os.path.splitext(os.path.basename(self.image_path))[0]  # Get image file name without extension
        annotation_file_path = os.path.join(self.save_directory, f"{file_name}_annotations")

        try:
            # Save annotations using selected format
            if format_selected == "YOLO":
                saver = YOLOAnnotationSaver()
                saver.save(self.bboxes, annotation_file_path + ".txt")
            elif format_selected == "COCO":
                saver = COCOAnnotationSaver()
                saver.save(self.bboxes, annotation_file_path + ".json")
            elif format_selected == "Pascal VOC":
                saver = PascalVOCAnnotationSaver()
                saver.save(self.bboxes, annotation_file_path + ".xml")

            messagebox.showinfo("Success", f"Annotations saved in {format_selected} format.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def load_next_image(self):
        # Clear canvas and annotations for the next image
        self.canvas.delete("all")
        self.bboxes = []
        self.rect_start = None
        self.rect_end = None
        self.rect_id = None
        self.load_image()

    def edit_annotations(self):
        # Reload the image without AI annotations
        self.bboxes = []  # Clear previous annotations
        self.rect_start = None
        self.rect_end = None
        self.rect_id = None
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Reload image without AI annotations (no pre-trained model)
        self.annotated_image = cv2.imread(self.image_path)  # Reload original image

        # Clear canvas and display image again
        self.canvas.delete("all")
        self.display_image()

        # Enable the "Edit" button after editing begins
        self.edit_button.config(state="disabled")

        # Enable undo and redo buttons
        self.undo_button.config(state="normal")
        self.redo_button.config(state="normal")

    def on_click(self, event):
        # Start drawing rectangle for bounding box
        if not self.rect_start:
            self.rect_start = (event.x, event.y)

    def on_drag(self, event):
        # Update the rectangle's endpoint as the user drags the mouse
        if self.rect_start:
            if self.rect_id:
                # Update existing rectangle
                self.canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], event.x, event.y)
            else:
                # Draw new rectangle as the user drags
                self.rect_id = self.canvas.create_rectangle(
                    self.rect_start[0], self.rect_start[1], event.x, event.y, outline="red", width=2
                )

    def on_release(self, event):
        # Finish drawing rectangle when mouse button is released
        if self.rect_start:
            self.rect_end = (event.x, event.y)
            self.bboxes.append((self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1]))
            self.undo_stack.append(("add", (self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1])))
            self.rect_start = None
            self.rect_end = None
            self.rect_id = None
            self.refresh_canvas()

    def undo(self):
        if self.undo_stack:
            action, bbox = self.undo_stack.pop()
            if action == "add":
                self.bboxes.remove(bbox)
                self.redo_stack.append(("add", bbox))

            self.refresh_canvas()

            # Disable undo if the stack is empty
            if not self.undo_stack:
                self.undo_button.config(state="disabled")

            # Enable redo button
            self.redo_button.config(state="normal")

    def redo(self):
        if self.redo_stack:
            action, bbox = self.redo_stack.pop()
            if action == "add":
                self.bboxes.append(bbox)
                self.undo_stack.append(("add", bbox))

            self.refresh_canvas()

            # Disable redo if the stack is empty
            if not self.redo_stack:
                self.redo_button.config(state="disabled")

            # Enable undo button
            self.undo_button.config(state="normal")

    def refresh_canvas(self):
        # Clear and redraw image with updated bounding boxes
        self.canvas.delete("all")
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)

    # Bind mouse events for creating bounding boxes
    app.canvas.bind("<Button-1>", app.on_click)  # On click, start drawing
    app.canvas.bind("<B1-Motion>", app.on_drag)  # While dragging, update rectangle
    app.canvas.bind("<ButtonRelease-1>", app.on_release)  # On release, finish drawing

    root.mainloop()
