"""
PhySynthTrainer - Physics-informed synthetic data training for event detection in solar radio bursts.
"""

from .burstGen import (
    generate_type_iii_burst,
    generate_many_random_t3_bursts,
    generate_type_2_burst,
    added_noise
)

from .utils import (
    plot_jpg_labeling,
    save_config_to_yml,
    load_config_from_yml,
    export_yolo_label,
    paint_arr_to_jpg,
    mask_to_bbox,
    mask_to_all_bboxes,
    visualize_mask_and_bboxes
)

from .densityModel import (
    saito77,
    leblanc98,
    parkerfit,
    newkirk
)

from .freqDrift import (
    freq_drift_f_t,
    freq_drift_t_f
)

__version__ = "0.1.0"
__author__ = "Peijin Zhang"
__email__ = "pz47@njit.edu"

def get_default_config():
    """Get the default configuration parameters.
    
    Returns:
        dict: Default configuration dictionary with standard parameters.
    """
    return {
        'freq_range': [30, 85],
        't_res': 0.5,
        't_start': 0.0,
        'N_freq': 640,
        'N_time': 640
    }
