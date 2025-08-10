#!/usr/bin/env python3
"""
PhySynthTrainer Comprehensive Test Script

This script combines the functionality of demo.py and test_config.py to provide
a comprehensive testing suite for all PhySynthTrainer features.

Run this script to test all features:
    python test.py
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
    added_noise,
    generate_type_2_burst
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

def test_configuration_management():
    """Test configuration saving and loading functionality."""
    print_section_header("1. Configuration Management Test")
    
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
        output_file='test_config.yml'
    )
    print(f"Configuration saved to: {config_file}")
    print()
    
    # Load configuration from YAML file
    loaded_config = load_config_from_yml('test_config.yml')
    print("Loaded configuration:")
    for key, value in loaded_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Use loaded configuration to generate a burst
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
    
    return loaded_config, img_bursts_loaded, mask_loaded, bbox_loaded

def test_single_burst_manual():
    """Test creating a single burst with deterministic parameters manually."""
    print_section_header("2. Single Burst Manual Generation Test")
    
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
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Single Type III Burst (Manual Generation)')
    plt.tight_layout()
    plt.savefig('test_single_burst_manual.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, t_ax, f_ax

def test_single_burst_function():
    """Test using the built-in function to generate a type III burst."""
    print_section_header("3. Single Burst Built-in Function Test")
    
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
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Type IIIb Radio Burst with Fine Structure and Bounding Box')
    plt.tight_layout()
    plt.savefig('test_single_burst_function.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, mask, bbox

def test_type_2_burst():
    """Test generating a type II radio burst."""
    print_section_header("4. Type II Burst Generation Test")
    
    # Generate type II burst
    img_bursts, mask, bboxes = generate_type_2_burst(
        freq_range=[30, 85],
        t_res=0.5,
        t_start=0.0,
        N_freq=640,
        N_time=640,
        v_shock=700,  # km/s
        t_s_start=80.0,
        t_s_end=180.0,
        eff_starting_freq=100,  # MHz
        laneNUM=6,
        harmonic_overlap=True
    )
    
    print(f"Generated type II burst image shape: {img_bursts.shape}")
    print(f"Bounding box (YOLO format): {bboxes}")
    print(f"Shock velocity: 700 km/s")
    print(f"Number of harmonic lanes: 6")
    print(f"Includes second harmonic: True")
    
    # Plot the result with bounding box
    plt.figure(figsize=(12, 8))
    plt.imshow(img_bursts.T, interpolation='nearest', aspect='auto', origin='lower')
    
    # Plot mask contour
    plt.contour(mask.T, levels=[0.5], colors='r', alpha=0.5)
    
        # Convert YOLO bbox to matplotlib format for plotting
    img_height, img_width = mask.shape
    
    if bboxes:
        for bbox in bboxes:
    
            x_center, y_center, width, height = bbox
        
            xmin = int((x_center - width / 2) * img_width)
            xmax = int((x_center + width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            ymax = int((y_center + height / 2) * img_height)    
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.title('Type II Radio Burst with Multiple Harmonic Lanes')
    plt.tight_layout()
    plt.savefig('test_type_2_burst.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, mask, bbox

def test_multiple_bursts():
    """Test generating multiple type III bursts."""
    print_section_header("4. Multiple Bursts Generation Test")
    
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
    plt.savefig('test_multiple_bursts.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts, bursts, is_t3b, f_ax

def test_add_noise(img_bursts, t_ax, f_ax):
    """Test adding background noise to the synthetic data."""
    print_section_header("5. Background Noise Addition Test")
    
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
    plt.savefig('test_bursts_with_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img_bursts_noisy

def test_yolo_export(bursts, is_t3b):
    """Test YOLO label export functionality."""
    print_section_header("6. YOLO Label Export Test")
    
    # Export YOLO labels for the multiple bursts
    print("Exporting YOLO labels...")
    label_file = export_yolo_label(
        bursts, 
        is_t3b, 
        output_dir='labels', 
        base_filename='test_bursts'
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

def test_save_image(img_bursts):
    """Test saving the generated image."""
    print_section_header("7. Image Saving Test")
    
    # Save the multiple bursts image
    print("Saving generated image...")
    paint_arr_to_jpg(img_bursts, 'test_bursts.jpg')
    print("Image saved as: test_bursts.jpg")
    
    return 'test_bursts.jpg'

def test_visualization_with_labels(image_file, label_file):
    """Test visualization with overlaid YOLO labels."""
    print_section_header("8. Visualization with Labels Test")
    
    # Plot with automatic configuration loading
    print("Plotting image with labels using package base.yml...")
    plot_jpg_labeling(image_file, label_file)
    
    # Plot with custom configuration
    print("\nPlotting image with labels using custom configuration...")
    plot_jpg_labeling(image_file, label_file, 'test_config.yml')
    
    print("Visualization test complete!")

def test_quick_config_demo():
    """Quick test of configuration functionality similar to test_config.py."""
    print_section_header("9. Quick Configuration Test")
    
    # Generate multiple bursts and create YOLO labels
    print("Generating multiple bursts for YOLO label demonstration...")
    img_bursts_multi, bursts_multi, is_t3b_multi = generate_many_random_t3_bursts(
        n_bursts=5,
        freq_range=[30, 85],
        t_res=0.5,
        t_start=0.0,
        N_freq=640,
        N_time=640
    )
    
    # Save image and labels
    # Save image
    paint_arr_to_jpg(img_bursts_multi, 'quick_test_bursts.jpg')
    print("Saved quick test image: quick_test_bursts.jpg")
    
    # Export YOLO labels
    label_file = export_yolo_label(bursts_multi, is_t3b_multi, base_filename='quick_test_bursts')
    print(f"Exported YOLO labels: {label_file}")
    print()
    
    # Demonstrate plotting with labels
    print("Demonstrating plot_jpg_labeling function...")
    if os.path.exists('quick_test_bursts.jpg') and os.path.exists('quick_test_bursts.txt'):
        print("Plotting image with YOLO labels...")
        plot_jpg_labeling('quick_test_bursts.jpg', 'quick_test_bursts.txt')
    else:
        print("Image or label files not found for plotting demonstration.")
    
    print("Quick configuration test completed successfully!")

def main():
    """Main test function that runs all tests."""
    print("PhySynthTrainer Comprehensive Test Script")
    print("=" * 60)
    print("This script tests all the key features of PhySynthTrainer")
    print("for generating synthetic solar radio bursts and preparing training data.")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('labels', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    try:
        # Run all tests
        print("\nStarting comprehensive testing...")
        
        # Test 1: Configuration management
        loaded_config, img_bursts_loaded, mask_loaded, bbox_loaded = test_configuration_management()
        
        # Test 2: Manual burst generation
        img_bursts_manual, t_ax, f_ax = test_single_burst_manual()
        
        # Test 3: Built-in function burst generation
        img_bursts_func, mask, bbox = test_single_burst_function()
        
        # Test 4: Type II burst generation
        img_bursts_type2, mask_type2, bbox_type2 = test_type_2_burst()
        
        # Test 5: Multiple bursts generation
        img_bursts_multi, bursts, is_t3b, f_ax = test_multiple_bursts()
        
        # Test 6: Noise addition
        img_bursts_noisy = test_add_noise(img_bursts_multi, t_ax, f_ax)
        
        # Test 7: YOLO export
        label_file = test_yolo_export(bursts, is_t3b)
        
        # Test 8: Image saving
        image_file = test_save_image(img_bursts_multi)
        
        # Test 9: Visualization with labels
        test_visualization_with_labels(image_file, label_file)
        
        # Test 10: Quick configuration demo
        test_quick_config_demo()
        
        print_section_header("All Tests Complete!")
        print("All PhySynthTrainer features have been tested successfully!")
        print("\nGenerated test files:")
        print("  - test_single_burst_manual.png")
        print("  - test_single_burst_function.png")
        print("  - test_type_2_burst.png")
        print("  - test_multiple_bursts.png")
        print("  - test_bursts_with_noise.png")
        print("  - test_bursts.jpg")
        print("  - test_config.yml")
        print("  - labels/test_bursts.txt")
        print("  - quick_test_bursts.jpg")
        print("  - quick_test_bursts.txt")
        print("\nThe test suite covers:")
        print("1. Configuration management with YAML")
        print("2. Single burst generation (manual)")
        print("3. Single burst generation (built-in function)")
        print("4. Type II burst generation")
        print("5. Multiple type III bursts generation")
        print("6. Background noise addition")
        print("7. YOLO label export for training")
        print("8. Image saving and visualization")
        print("9. Reproducible experiments")
        print("10. Quick configuration testing")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Please check that all dependencies are installed and the package is properly set up.")
        raise

if __name__ == "__main__":
    main()
