#!/usr/bin/env python3
"""
Test script to demonstrate configuration saving and loading functionality.
"""

from physynthtrainer.utils import save_config_to_yml, load_config_from_yml, plot_jpg_labeling
from physynthtrainer.burstGen import generate_type_iii_burst, generate_many_random_t3_bursts
import os

def main():
    # Example configuration parameters
    freq_range = [30, 85]  # MHz
    t_res = 0.5  # sec
    t_start = 0.0  # sec
    N_freq = 640
    N_time = 640
    
    print("Original configuration:")
    print(f"  freq_range: {freq_range}")
    print(f"  t_res: {t_res}")
    print(f"  t_start: {t_start}")
    print(f"  N_freq: {N_freq}")
    print(f"  N_time: {N_time}")
    print()
    
    # Save configuration to YAML file
    config_file = save_config_to_yml(
        freq_range=freq_range,
        t_res=t_res,
        t_start=t_start,
        N_freq=N_freq,
        N_time=N_time,
        output_file='burst_config.yml'
    )
    print(f"Configuration saved to: {config_file}")
    print()
    
    # Load configuration from YAML file
    loaded_config = load_config_from_yml('burst_config.yml')
    print("Loaded configuration:")
    for key, value in loaded_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Use loaded configuration to generate bursts
    print("Generating burst with loaded configuration...")
    img_bursts_loaded, mask_loaded, bbox_loaded = generate_type_iii_burst(
        freq_range=loaded_config['freq_range'],
        t_res=loaded_config['t_res'],
        t_start=loaded_config['t_start'],
        N_freq=loaded_config['N_freq'],
        N_time=loaded_config['N_time'],
        v_beam=0.15,
        t0_burst=80.0,
        decay_t_dur_100Mhz=1.0,
        Burst_intensity=1.0,
        burststarting_freq=40.0,
        burstending_freq=60.0,
        edge_freq_ratio=0.05,
        fine_structure=True
    )
    
    print(f"Generated burst with loaded config - Image shape: {img_bursts_loaded.shape}")
    print(f"Bounding box: {bbox_loaded}")
    print()
    
    # Generate multiple bursts and create YOLO labels
    print("Generating multiple bursts for YOLO label demonstration...")
    img_bursts_multi, bursts_multi, is_t3b_multi = generate_many_random_t3_bursts(
        n_bursts=5,
        freq_range=loaded_config['freq_range'],
        t_res=loaded_config['t_res'],
        t_start=loaded_config['t_start'],
        N_freq=loaded_config['N_freq'],
        N_time=loaded_config['N_time']
    )
    
    # Save image and labels
    from physynthtrainer.utils import paint_arr_to_jpg, export_yolo_label
    
    # Save image
    paint_arr_to_jpg(img_bursts_multi, 'test_bursts.jpg')
    print("Saved test image: test_bursts.jpg")
    
    # Export YOLO labels
    label_file = export_yolo_label(bursts_multi, is_t3b_multi, base_filename='test_bursts')
    print(f"Exported YOLO labels: {label_file}")
    print()
    
    # Demonstrate plotting with labels
    print("Demonstrating plot_jpg_labeling function...")
    if os.path.exists('test_bursts.jpg') and os.path.exists('test_bursts.txt'):
        print("Plotting image with YOLO labels...")
        plot_jpg_labeling('test_bursts.jpg', 'test_bursts.txt')
    else:
        print("Image or label files not found for plotting demonstration.")
    
    print()
    print("Configuration test completed successfully!")

if __name__ == "__main__":
    main()
