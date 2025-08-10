#!/usr/bin/env python3
"""
Generate YOLO Dataset Script

This script generates a synthetic dataset of radio bursts for training YOLO object detection models.
It creates images with multiple type III radio bursts and corresponding YOLO format labels.

Usage:
    python gen_yolo_dataset.py [--total-set TOTAL_SET] [--output-dir OUTPUT_DIR]

Example:
    python gen_yolo_dataset.py --total-set 6000 --output-dir ./dataset
"""

import os
import gc
import argparse
import numpy as np
from physynthtrainer.burstGen import generate_many_random_t3_bursts, added_noise
from physynthtrainer.utils import paint_arr_to_jpg


def generate_yolo_dataset(total_set=6000, output_dir='./dataset'):
    """
    Generate a YOLO dataset with synthetic radio bursts.
    
    Args:
        total_set (int): Total number of images to generate. Defaults to 6000.
        output_dir (str): Output directory for the dataset. Defaults to './dataset'.
    """
    # Clean up memory
    gc.collect()
    
    # Setup directories
    dir_labels = os.path.join(output_dir, 'labels')
    dir_images = os.path.join(output_dir, 'images')
    
    # Create directories if they don't exist
    os.makedirs(dir_labels, exist_ok=True)
    os.makedirs(dir_images, exist_ok=True)
    
    print(f"Generating {total_set} images in {output_dir}")
    print(f"Labels will be saved to: {dir_labels}")
    print(f"Images will be saved to: {dir_images}")
    
    # Generate dataset
    for i in range(total_set):
        if i % 100 == 0:
            print(f"Progress: {i}/{total_set} ({(i/total_set)*100:.1f}%)")
        
        # Generate filename with zero-padding
        fname = f'b{i:05d}'
        
        # Random number of bursts between 5 and 60
        num_bursts = np.random.randint(5, 60)
        
        # Generate bursts
        img_bursts, bursts, t3b = generate_many_random_t3_bursts(n_bursts=num_bursts)
        
        # Add noise and background
        interpolated_norm = added_noise(None, None, 0.2, noise_size=[32, 8])
        y = np.linspace(0.1, 1.2, 640)
        const_bg = np.tile(y[:, np.newaxis], (1, 640))
        img_bursts_withbg = img_bursts + interpolated_norm.T + const_bg.T
        
        # Write YOLO labels
        label_file = os.path.join(dir_labels, fname + '.txt')
        with open(label_file, 'w') as f:
            for is_t3b_this, bbox in zip(t3b, bursts):
                # Class label: 1 for type IIIb, 0 for type III
                if is_t3b_this:
                    f.write('1 ')
                else:
                    f.write('0 ')
                
                # Bounding box coordinates in YOLO format
                x_center, y_center, width, height = bbox
                f.write(f'{x_center} {y_center} {width} {height}\n')
        
        # Save image
        image_file = os.path.join(dir_images, fname + '.jpg')
        paint_arr_to_jpg(img_bursts_withbg, image_file)
        
        # Clean up memory periodically
        if i % 500 == 0:
            gc.collect()
    
    print(f"Dataset generation complete!")
    print(f"Generated {total_set} images and labels in {output_dir}")
    print(f"Class labels: 0 = Type III, 1 = Type IIIb")


def main():
    """Main function to parse arguments and run dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate YOLO dataset with synthetic radio bursts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--total-set', 
        type=int, 
        default=6000,
        help='Total number of images to generate'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./dataset',
        help='Output directory for the dataset'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.total_set <= 0:
        parser.error("total-set must be a positive integer")
    
    if not os.path.exists(args.output_dir) and not os.access(os.path.dirname(args.output_dir), os.W_OK):
        parser.error(f"Cannot create output directory: {args.output_dir}")
    
    try:
        generate_yolo_dataset(args.total_set, args.output_dir)
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user")
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        raise


if __name__ == "__main__":
    main()
