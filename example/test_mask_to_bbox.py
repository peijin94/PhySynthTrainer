#!/usr/bin/env python3
"""
Test script for the new mask_to_bbox functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from physynthtrainer.utils import mask_to_bbox, mask_to_all_bboxes, visualize_mask_and_bboxes

def test_simple_mask():
    """Test with a simple rectangular mask."""
    print("Testing simple rectangular mask...")
    
    # Create a simple rectangular mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    
    # Get bounding box
    bbox = mask_to_bbox(mask, min_area=50)
    print(f"Single bbox: {bbox}")
    
    # Get all bounding boxes
    all_bboxes = mask_to_all_bboxes(mask, min_area=50)
    print(f"All bboxes: {all_bboxes}")
    
    # Visualize
    visualize_mask_and_bboxes(mask, all_bboxes, "Simple Rectangular Mask")

def test_multiple_components():
    """Test with multiple disconnected components."""
    print("\nTesting multiple disconnected components...")
    
    # Create mask with multiple components
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:90] = 1  # Small square
    mask[60:90, 60:90] = 1  # Large square
    mask[40:50, 40:50] = 1  # Medium square
    
    # Get all bounding boxes
    all_bboxes = mask_to_all_bboxes(mask, min_area=50)
    print(f"All bboxes (sorted by area): {all_bboxes}")
    
    # Visualize
    visualize_mask_and_bboxes(mask, all_bboxes, "Multiple Components")

def test_complex_mask():
    """Test with a more complex mask."""
    print("\nTesting complex mask...")
    
    # Create a complex mask with irregular shapes
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some irregular shapes
    y, x = np.ogrid[:100, :100]
    
    # Circle
    mask[(x-25)**2 + (y-25)**2 <= 15**2] = 1
    
    # Rectangle
    mask[60:80, 60:80] = 1
    
    # L-shaped region
    mask[10:40, 10:20] = 1
    mask[30:40, 10:40] = 1
    
    # Get all bounding boxes
    all_bboxes = mask_to_all_bboxes(mask, min_area=100)
    print(f"Complex mask bboxes: {all_bboxes}")
    
    # Visualize
    visualize_mask_and_bboxes(mask, all_bboxes, "Complex Mask")

def test_empty_mask():
    """Test with empty mask."""
    print("\nTesting empty mask...")
    
    mask = np.zeros((50, 50), dtype=np.uint8)
    
    bbox = mask_to_bbox(mask, min_area=10)
    print(f"Empty mask bbox: {bbox}")
    
    all_bboxes = mask_to_all_bboxes(mask, min_area=10)
    print(f"Empty mask all bboxes: {all_bboxes}")

def test_small_components():
    """Test with components below area threshold."""
    print("\nTesting small components below threshold...")
    
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:15, 10:15] = 1  # 5x5 = 25 pixels
    
    # Test with threshold above component size
    bbox = mask_to_bbox(mask, min_area=30)
    print(f"Small component with high threshold: {bbox}")
    
    # Test with threshold below component size
    bbox = mask_to_bbox(mask, min_area=20)
    print(f"Small component with low threshold: {bbox}")

if __name__ == "__main__":
    print("Testing mask_to_bbox functions...")
    print("=" * 50)
    
    test_simple_mask()
    test_multiple_components()
    test_complex_mask()
    test_empty_mask()
    test_small_components()
    
    print("\nAll tests completed!")
