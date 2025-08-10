# PhySynthTrainer

Physics-informed synthetic data training for event detection in solar radio bursts.

PhySynthTrainer is a Python library for generating synthetic type III solar radio bursts. This tool is designed to create realistic-looking radio burst data that can be used to train machine learning models for event detection. The generation process is physics-informed, utilizing various density models of the solar corona to simulate the frequency drift of the bursts.

## Features

- **Generate Type II, Type III and Type IIIb Bursts:** Create type II bursts with multiple harmonic lanes, standard type III bursts, or type IIIb bursts with fine structures.
- **Customizable Parameters:** Adjust various parameters such as beam velocity, shock velocity, burst intensity, frequency range, and more.
- **Multiple Density Models:** Supports various solar corona density models including Saito, Leblanc, Parker, and Newkirk.
- **Bounding Box Generation:** Automatically generates bounding boxes for the bursts, suitable for training object detection models.
- **Background Noise:** Ability to add background noise to the synthetic data to create more realistic training samples.
- **Configuration Management:** Save and load burst generation parameters in YAML format for reproducible experiments.
- **YOLO Label Export:** Export bounding box annotations in YOLO format for training object detection models.
- **Visualization:** Plot images with overlaid YOLO labels for inspection and analysis.

## Installation

Install PhySynthTrainer directly from the GitHub repository:

```bash
pip install git+https://github.com/peijin94/PhySynthTrainer.git
```

## Usage

Here is a simple example of how to generate a type III radio burst and save it as an image:

```python
import numpy as np
from physynthtrainer.burstGen import generate_type_iii_burst
from physynthtrainer.utils import plot_and_save_burst

# Generate a single type III burst
img_bursts, mask, bbox = generate_type_iii_burst(
    fine_structure=True
)

# Save the burst image with its bounding box
plot_and_save_burst(img_bursts, bbox, True, filename='type3b_burst.jpg')
```

This will generate an image named `type3b_burst.jpg` in your current directory, showing the synthetic radio burst with its bounding box.

### Configuration Management

Save and load burst generation parameters for reproducible experiments:

```python
from physynthtrainer.utils import save_config_to_yml, load_config_from_yml

# Save configuration
save_config_to_yml(
    freq_range=[30, 85],
    t_res=0.5,
    t_start=0.0,
    N_freq=640,
    N_time=640,
    output_file='my_config.yml'
)

# Load configuration
config = load_config_from_yml('my_config.yml')
print(config['freq_range'])  # [30, 85]
```

### YOLO Label Export

Export bounding box annotations for training object detection models:

```python
from physynthtrainer.utils import export_yolo_label
from physynthtrainer.burstGen import generate_many_random_t3_bursts

# Generate multiple bursts
img_bursts, bursts, is_t3b = generate_many_random_t3_bursts(n_bursts=10)

# Export YOLO labels
export_yolo_label(bursts, is_t3b, output_dir='labels', base_filename='bursts')
```

### Visualization with Labels

Plot images with overlaid YOLO labels for inspection:

```python
from physynthtrainer.utils import plot_jpg_labeling

# Plot image with labels (automatically loads package configuration or uses defaults)
plot_jpg_labeling('burst.jpg', 'burst.txt')

# Or specify a custom configuration file
plot_jpg_labeling('burst.jpg', 'burst.txt', 'my_config.yml')
```

### Mask to Bounding Box Conversion

Convert binary masks to YOLO format bounding boxes:

```python
from physynthtrainer.utils import mask_to_bbox, mask_to_all_bboxes, visualize_mask_and_bboxes

# Convert mask to single largest bounding box
bbox = mask_to_bbox(mask, min_area=20)

# Convert mask to all bounding boxes above threshold
all_bboxes = mask_to_all_bboxes(mask, min_area=20)

# Visualize mask with bounding boxes
visualize_mask_and_bboxes(mask, all_bboxes, "My Mask")
```

## Contributing

Contributions are welcome! If you have any suggestions or find a bug, please open an issue on the [GitHub issue tracker](https://github.com/peijin94/PhySynthTrainer/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Type Definition

This table defines the type of the data that is used in the training process.

| type_id | Name | Description |
|---|---|---|
| 0 | t3 | Type III radio burst |
| 1 | t3b | Type III radio burst with fine structures |
| 2 | t2 | Type II radio burst |