#!/usr/bin/env python3
"""
PhySynthTrainer Demo Script

This script demonstrates the key features of PhySynthTrainer for generating 
synthetic solar radio bursts and preparing training data for machine learning models.

Run this script to see all features in action:
    python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from physynthtrainer.burstGen import (
    generate_quasi_periodic_signal, 
    biGaussian, 
    generate_type_iii_burst, 
    generate_many_random_t3_bursts, 
    create_radio_burst_hash_table,
    added_noise
)
from physynthtrainer.utils import (
    paint_arr_to_jpg, 
    export_yolo_label, 
    plot_jpg_labeling,
    save_config_to_yml, 
    load_config_from_yml
)
from physynthtrainer import freqDrift

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def demo_single_burst_manual():
    """Demonstrate creating a single burst with deterministic parameters manually."""
    print_section_header("1. Create Single Burst with Deterministic Parameters")
    
    # Parameters
    freq_range = [30, 85]  # MHz
    t_res = 0.5            # sec
    t_start = 0.0          # sec
    N_freq = 640
    N_time = 640
    
    # Burst parameters
    v_beam = 0.15          # c
    t0_burst = 80.0        # sec
    decay_t_dur_100Mhz = 1 # sec
    Burst_intensity = 1.0
    burststarting_freq = 40.0
    burstending_freq = 60.0
    edge_freq_ratio = 0.01
    fine_structure = True
    
    print(f"Creating burst with parameters:")
    print(f"  Frequency range: {freq_range} MHz")
    print(f"  Time resolution: {t_res} sec")
    print(f"  Beam velocity: {v_beam}c")
    print(f"  Fine structure: {fine_structure}")
    
    # Create arrays
    img_bursts = np.zeros((N_time, N_freq))
    t_ax = t_start + np.arange(0, N_time) * t_res
    f_ax = np.linspace(freq_range[0], freq_range[1], N_freq)
    
    edge_freq = (burstending_freq - burststarting_freq) * edge_freq_ratio
    t_burst = []
    
    # Add fine structure modulation if enabled
    if fine_structure:
        freq_modulation = generate_quasi_periodic_signal(
            t_arr=f_ax, base_freq=0.5, num_harmonics=5, 
            noise_level=0, freqvar=0.1
        )
        # normalize to 0-1
        freq_modulation = (freq_modulation - np.min(freq_modulation)) / (np.max(freq_modulation) - np.min(freq_modulation))
    
    # Generate burst for each frequency
    for pix_f, f_this in enumerate(f_ax):
        t_burst.append(freqDrift.freq_drift_t_f(f_ax[pix_f], v_beam, t0_burst))
        decay_t_dur_this = decay_t_dur_100Mhz * (100.0/f_this)
        burst_amp = Burst_intensity * (np.tanh((f_this - burststarting_freq) / edge_freq)+1) * (np.tanh((burstending_freq - f_this) / edge_freq)+1)
        burst_amp = burst_amp * freq_modulation[pix_f] if fine_structure else burst_amp
        l_curve = biGaussian(t_ax, t_burst[-1], decay_t_dur_this, decay_t_dur_this/2, burst_amp)
        img_bursts[:, pix_f] += l_curve
    
    t_burst = np.array(t_burst)
    
    print(f"Generated burst image shape: {img_bursts.shape}")
    print(f"Burst time range: {t_burst.min():.1f} to {t_burst.max():.1f} seconds")
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.imshow(img_bursts.T, interpolation='nearest', aspect='auto', origin='lower', 
               extent=[t_start, t_start + N_time * t_res, freq_range[0], freq_range[1]])
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Single Type III Burst (Manual Generation)')
    plt.tight_layout()
    plt.savefig('demo_single_burst_manual.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, t_ax, f_ax

def demo_single_burst_function():
    """Demonstrate using the built-in function to generate a type III burst."""
    print_section_header("2. Generate Single Burst Using Built-in Function")
    
    # Generate burst using built-in function
    img_bursts, mask, bbox = generate_type_iii_burst(
        freq_range=[30, 85], 
        t_res=0.5, 
        t_start=0.0, 
        N_freq=640, 
        N_time=640,
        v_beam=0.15, 
        t0_burst=80.0, 
        decay_t_dur_100Mhz=1.0,
        Burst_intensity=1.0, 
        burststarting_freq=40.0,
        burstending_freq=60.0, 
        edge_freq_ratio=0.05,
        fine_structure=True
    )
    
    print(f"Generated burst image shape: {img_bursts.shape}")
    print(f"Bounding box (YOLO format): {bbox}")
    print(f"Has fine structure (t3b): {True}")
    
    # Plot the result with bounding box
    plt.figure(figsize=(10, 8))
    plt.imshow(img_bursts.T, interpolation='nearest', aspect='auto', origin='lower')
    
    # Plot mask and bbox
    plt.contour(mask.T, levels=[0.5], colors='r', alpha=0.5)
    
    # Convert YOLO bbox to matplotlib format
    img_height, img_width = mask.shape
    x_center, y_center, width, height = bbox
    
    xmin = int((x_center - width / 2) * img_width)
    xmax = int((x_center + width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    ymax = int((y_center + height / 2) * img_height)
    
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-', linewidth=2)
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Type IIIb Radio Burst with Fine Structure and Bounding Box')
    plt.tight_layout()
    plt.savefig('demo_single_burst_function.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, mask, bbox

def demo_multiple_bursts():
    """Demonstrate generating multiple type III bursts."""
    print_section_header("3. Generate Multiple Type III Bursts")
    
    # Create hash table for faster generation
    print("Creating hash table for faster generation...")
    t_burst_hash, f_ax, v_ax = create_radio_burst_hash_table(
        freq_range=[30, 85], 
        N_freq=640, 
        v_range=[0.05, 0.5], 
        N_v=200
    )
    
    # Generate multiple bursts
    print("Generating multiple bursts...")
    img_bursts, bursts, is_t3b = generate_many_random_t3_bursts(
        n_bursts=30, 
        use_hash_table=True, 
        freq_range=[35, 80],
        hash_table=t_burst_hash, 
        v_hash=v_ax
    )
    
    print(f"Generated {len(bursts)} bursts")
    print(f"Image shape: {img_bursts.shape}")
    print(f"Number of t3b (fine structure): {sum(is_t3b)}")
    print(f"Number of t3 (no fine structure): {len(is_t3b) - sum(is_t3b)}")
    
    # Plot the result
    plt.figure(figsize=(12, 8))
    plt.imshow(img_bursts.T, interpolation='nearest', aspect='auto', origin='lower',
               extent=[0, 200, 30, 85])
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Multiple Type III Radio Bursts')
    plt.tight_layout()
    plt.savefig('demo_multiple_bursts.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, bursts, is_t3b, f_ax

def demo_add_noise(img_bursts, t_ax, f_ax):
    """Demonstrate adding background noise to the synthetic data."""
    print_section_header("4. Add Background Noise")
    
    # Add noise
    print("Adding background noise...")
    noise_bg = added_noise(t_ax, f_ax, noise_size=[32,8], noise_level=0.35)
    
    # Add constant background
    y = np.linspace(0.1, 1, 640)
    const_bg = np.tile(y[:, np.newaxis], (1, 640))
    
    img_bursts_noisy = img_bursts + noise_bg.T + const_bg.T
    
    print(f"Added noise with level: 0.35")
    print(f"Added constant background gradient")
    
    # Plot the result
    plt.figure(figsize=(12, 8))
    plt.imshow(img_bursts_noisy.T, interpolation='nearest', aspect='auto', origin='lower',
               extent=[0, 200, 30, 85])
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Multiple Bursts with Background Noise')
    plt.tight_layout()
    plt.savefig('demo_bursts_with_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts_noisy

def demo_configuration_management():
    """Demonstrate configuration management features."""
    print_section_header("5. Configuration Management")
    
    # Save configuration to YAML file
    print("Saving configuration to YAML file...")
    config_file = save_config_to_yml(
        freq_range=[30, 85],
        t_res=0.5,
        t_start=0.0,
        N_freq=640,
        N_time=640,
        output_file='demo_config.yml'
    )
    
    print(f"Configuration saved to: {config_file}")
    
    # Load configuration from YAML file
    print("\nLoading configuration from YAML file...")
    loaded_config = load_config_from_yml('demo_config.yml')
    
    print("Loaded configuration:")
    for key, value in loaded_config.items():
        print(f"  {key}: {value}")
    
    # Use loaded configuration to generate a burst
    print("\nGenerating burst with loaded configuration...")
    img_bursts_loaded, mask_loaded, bbox_loaded = generate_type_iii_burst(
        freq_range=loaded_config['freq_range'],
        t_res=loaded_config['t_res'],
        t_start=loaded_config['t_start'],
        N_freq=loaded_config['N_freq'],
        N_time=loaded_config['N_time'],
        fine_structure=False  # Generate t3 (no fine structure)
    )
    
    print(f"Generated burst with loaded config - Image shape: {img_bursts_loaded.shape}")
    print(f"Bounding box: {bbox_loaded}")
    
    return loaded_config, img_bursts_loaded, mask_loaded, bbox_loaded

def demo_yolo_export(bursts, is_t3b):
    """Demonstrate YOLO label export functionality."""
    print_section_header("6. YOLO Label Export")
    
    # Export YOLO labels for the multiple bursts
    print("Exporting YOLO labels...")
    label_file = export_yolo_label(
        bursts, 
        is_t3b, 
        output_dir='labels', 
        base_filename='demo_bursts'
    )
    
    print(f"YOLO labels exported to: {label_file}")
    
    # Display the first few labels
    with open(label_file, 'r') as f:
        lines = f.readlines()
        print(f"\nFirst 5 labels (class x_center y_center width height):")
        for i, line in enumerate(lines[:5]):
            class_id = int(line.split()[0])
            class_name = "t3b" if class_id == 1 else "t3"
            print(f"  {i+1}: {line.strip()} ({class_name})")
    
    return label_file

def demo_save_image(img_bursts):
    """Demonstrate saving the generated image."""
    print_section_header("7. Save Generated Image")
    
    # Save the multiple bursts image
    print("Saving generated image...")
    paint_arr_to_jpg(img_bursts, 'demo_bursts.jpg')
    print("Image saved as: demo_bursts.jpg")
    
    return 'demo_bursts.jpg'

def demo_visualization_with_labels(image_file, label_file):
    """Demonstrate visualization with overlaid YOLO labels."""
    print_section_header("8. Visualization with Labels")
    
    # Plot with automatic configuration loading
    print("Plotting image with labels using package base.yml...")
    plot_jpg_labeling(image_file, label_file)
    
    # Plot with custom configuration
    print("\nPlotting image with labels using custom configuration...")
    plot_jpg_labeling(image_file, label_file, 'demo_config.yml')
    
    print("Visualization complete!")

def main():
    """Main demo function that runs all demonstrations."""
    print("PhySynthTrainer Demo Script")
    print("=" * 60)
    print("This script demonstrates all the key features of PhySynthTrainer")
    print("for generating synthetic solar radio bursts and preparing training data.")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('labels', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    try:
        # Run all demos
        img_bursts_manual, t_ax, f_ax = demo_single_burst_manual()
        
        img_bursts_func, mask, bbox = demo_single_burst_function()
        
        img_bursts_multi, bursts, is_t3b, f_ax = demo_multiple_bursts()
        
        img_bursts_noisy = demo_add_noise(img_bursts_multi, t_ax, f_ax)
        
        loaded_config, img_bursts_loaded, mask_loaded, bbox_loaded = demo_configuration_management()
        
        label_file = demo_yolo_export(bursts, is_t3b)
        
        image_file = demo_save_image(img_bursts_multi)
        
        demo_visualization_with_labels(image_file, label_file)
        
        print_section_header("Demo Complete!")
        print("All PhySynthTrainer features have been demonstrated successfully!")
        print("\nGenerated files:")
        print("  - demo_single_burst_manual.png")
        print("  - demo_single_burst_function.png")
        print("  - demo_multiple_bursts.png")
        print("  - demo_bursts_with_noise.png")
        print("  - demo_bursts.jpg")
        print("  - demo_config.yml")
        print("  - labels/demo_bursts.txt")
        print("\nThe demo showcases:")
        print("1. Single and multiple burst generation")
        print("2. Background noise addition")
        print("3. Configuration management with YAML")
        print("4. YOLO label export for training")
        print("5. Image saving and visualization")
        print("6. Reproducible experiments")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please check that all dependencies are installed and the package is properly set up.")
        raise

if __name__ == "__main__":
    main()
