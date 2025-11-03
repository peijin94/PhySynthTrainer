# Scripts Directory

This directory contains utility scripts for the PhySynthTrainer project.

## gen_yolo_dataset.py

A script to generate synthetic YOLO datasets for training object detection models on radio burst data.

### Features

- Generates synthetic type III and type IIIb radio bursts
- Creates corresponding YOLO format labels
- Adds realistic noise and background
- Configurable dataset size and output directory
- Progress tracking and memory management

### Usage

```bash
# Basic usage (generates 6000 images in ./dataset)
python scripts/gen_yolo_dataset.py

# Custom dataset size
python scripts/gen_yolo_dataset.py --total-set 1000

# Custom output directory
python scripts/gen_yolo_dataset.py --output-dir ./my_dataset

# Both custom parameters
python scripts/gen_yolo_dataset.py --total-set 2000 --output-dir ./custom_dataset
```

### Output Structure

The script creates the following directory structure:

```
output_dir/
├── images/
│   ├── b00000.jpg
│   ├── b00001.jpg
│   └── ...
└── labels/
    ├── b00000.txt
    ├── b00001.txt
    └── ...
```

### Label Format

Each label file contains YOLO format annotations:

```
class_id center_x center_y width height
```

Where:
- `class_id`: 0 for Type III, 1 for Type IIIb
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized dimensions (0-1)

### Requirements

- Python 3.6+
- numpy
- physynthtrainer package
- matplotlib (for image saving)

### Notes

- The script generates between 5-60 bursts per image randomly
- Memory is cleaned up every 500 images to prevent memory issues
- Progress is displayed every 100 images
- The script can be interrupted with Ctrl+C and will exit gracefully
