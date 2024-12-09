# Automatic Image Annotation Tool

This repository contains a Python-based application for automatic annotation or labeling of images using deep learning models. It provides a graphical user interface (GUI) for loading, annotating, and saving image annotations in multiple formats, including YOLO, Pascal VOC, and COCO.

## Features
- **AI-Powered Annotation**: Uses pre-trained models such as Faster R-CNN and YOLO for object detection and automatic annotation.
- **User-Friendly GUI**: Built with `tkinter`, allowing users to load, view, and manage annotations interactively.
- **Multi-Format Support**: Save annotations in YOLO, Pascal VOC, or COCO formats.
- **Flexible File Organization**: Supports structured folder hierarchies for input images and saves outputs in corresponding directories.

## Installation

### Prerequisites
1. Python 3.7 or higher.
2. Required Python libraries:
    - `torch`
    - `torchvision`
    - `opencv-python`
    - `Pillow`
    - `tkinter` (comes pre-installed with Python)
    - `numpy`

Install the dependencies using `pip`:
```bash
pip install torch torchvision opencv-python Pillow numpy
```

### Clone the Repository
```bash
git clone https://github.com/IMApurbo/ai-labeler.git
cd ai-labeler
```

## Usage

### 1. **Run the GUI Application**
Execute the `gui.py` script:
```bash
cd gui
python gui.py
```

### 2. **Run the All-in-One Script**
Alternatively, use the standalone script that combines all functionalities into a single file:
```bash
python ai-labeler(all-in-one).py
```

### 3. **Load and Annotate Images**
- Launch the application.
- Use the "Load Image" button to load an image.
- The application automatically detects objects and displays bounding boxes.
- Save annotations in the desired format by selecting it from the dropdown menu and clicking "Save Annotations."

### 4. **Supported Formats**
- YOLO: `.txt`
- Pascal VOC: `.xml`
- COCO: `.json`

## Folder Structure
```
ai-labeler/
├── ai-labeler/           
│   ├── __init__.py       
│   ├── annotation_saver.py
│   ├── annotator.py
│   ├── labeler.py
│   ├── model_loader.py   
│   └── utils.py     
├── gui.py
├── ai-labeler(all-in-one).py       
├── README.md        
└── requirements.txt   
```

## Screenshots
Add screenshots of your GUI in action to help users visualize its functionality.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
