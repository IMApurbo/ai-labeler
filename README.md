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
ai-labeler/                  # Root directory of the AI Labeler project
├── ai-labeler/              # Core package containing the main modules
│   ├── __init__.py          # Package initializer for the `ai-labeler` module
│   ├── annotation_saver.py  # Handles saving annotations in the desired format
│   ├── annotator.py         # Main annotation logic for labeling images or data
│   ├── labeler.py           # Core labeler class to interface with the GUI and model
│   ├── model_loader.py      # Responsible for loading and managing AI models
│   └── utils.py             # Utility functions for file handling, image processing, etc.
├── gui.py                   # Script to create and manage the graphical user interface (GUI)
├── ai-labeler(all-in-one).py # All-in-one script combining all modules for standalone execution
├── README.md                # Documentation file explaining the project's purpose, usage, and setup
└── requirements.txt         # List of dependencies required to run the project

```

## Screenshots
Add screenshots of your GUI in action to help users visualize its functionality.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
