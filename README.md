# PhySynthTrainer

Physics-informed synthetic data training for event detection in solar radio bursts.

PhySynthTrainer is a Python library for generating synthetic type III solar radio bursts. This tool is designed to create realistic-looking radio burst data that can be used to train machine learning models for event detection. The generation process is physics-informed, utilizing various density models of the solar corona to simulate the frequency drift of the bursts.

## Features

- **Generate Type III and Type IIIb Bursts:** Create standard type III bursts or type IIIb bursts with fine structures.
- **Customizable Parameters:** Adjust various parameters such as beam velocity, burst intensity, frequency range, and more.
- **Multiple Density Models:** Supports various solar corona density models including Saito, Leblanc, Parker, and Newkirk.
- **Bounding Box Generation:** Automatically generates bounding boxes for the bursts, suitable for training object detection models.
- **Background Noise:** Ability to add background noise to the synthetic data to create more realistic training samples.

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