
from PIL import Image
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict, Any
import yaml
import os


def paint_arr_to_jpg(arr, filename='test.jpg', do_norm=True):
    """Saves a 2D numpy array as a jpg image.

    Args:
        arr (np.ndarray): The 2D array to save.
        filename (str, optional): The name of the file to save. 
            Defaults to 'test.jpg'.
        do_norm (bool, optional): Whether to normalize the array before saving. 
            Defaults to True.
    """
    norm = mcolors.Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = plt.get_cmap('CMRmap') 
    img = cmap(norm(arr.T))
    imgsave = (img * 255).astype(np.uint8)[:,:,0:3]
    im = Image.fromarray(imgsave)
    im.save(filename)


def plot_jpg_labeling(img_file: str, labeling_txt: str, config: Union[str, Dict[str, Any]] = None) -> None:
    """Plot an image with YOLO format labels overlaid.
    
    This function loads an image file and overlays bounding box annotations from a YOLO label file.
    It automatically loads configuration parameters to set the correct axis labels and ranges.
    
    Args:
        img_file (str): Path to the image file (.jpg, .png, etc.)
        labeling_txt (str): Path to the YOLO label file (.txt)
        config (Union[str, Dict[str, Any]], optional): Configuration file path or dictionary. 
            If None, loads from package base.yml. Defaults to None.
    
    Raises:
        FileNotFoundError: If image file or label file doesn't exist.
        ValueError: If label file format is invalid.
    
    Example:
        >>> plot_jpg_labeling('burst.jpg', 'burst.txt')
        >>> plot_jpg_labeling('burst.jpg', 'burst.txt', 'my_config.yml')
    """
    import matplotlib.patches as patches
    
    # Load configuration
    if config is None:
        # Try to load from package base.yml using importlib.resources
        try:
            import importlib.resources as pkg_resources
            config_file = pkg_resources.files('physynthtrainer') / 'base.yml'
            if config_file.exists():
                config = load_config_from_yml(str(config_file))
            else:
                # Fallback to default configuration
                try:
                    from . import get_default_config
                    config = get_default_config()
                except ImportError:
                    config = {
                        'freq_range': [30, 85],
                        't_res': 0.5,
                        't_start': 0.0,
                        'N_freq': 640,
                        'N_time': 640
                    }
                print("Warning: Package base.yml not found, using default configuration.")
        except Exception as e:
            # Fallback to default configuration
            try:
                from . import get_default_config
                config = get_default_config()
            except ImportError:
                config = {
                    'freq_range': [30, 85],
                    't_res': 0.5,
                    't_start': 0.0,
                    'N_freq': 640,
                    'N_time': 640
                }
            print(f"Warning: Could not load package configuration: {e}")
            print("Using default configuration.")
    elif isinstance(config, str):
        # Load from specified config file
        config = load_config_from_yml(config)
    # If config is already a dict, use it directly
    
    # Check if files exist
    if not os.path.exists(img_file):
        raise FileNotFoundError(f"Image file not found: {img_file}")
    if not os.path.exists(labeling_txt):
        raise FileNotFoundError(f"Label file not found: {labeling_txt}")
    
    # Load image
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Get image dimensions
    img_height, img_width = img_array.shape[:2]
    
    # Create figure
    plt.figure()
    
    # Get configuration parameters for axis scaling
    freq_range = config.get('freq_range', [30, 85])
    t_res = config.get('t_res', 0.5)
    t_start = config.get('t_start', 0.0)
    N_time = config.get('N_time', 640)
    
    # Calculate time and frequency ranges
    time_range = [t_start, t_start + N_time * t_res]
    
    # Plot image with proper extent to match time and frequency axes
    plt.imshow(img_array, interpolation='nearest', aspect='auto', origin='lower',
               extent=[time_range[0], time_range[1], freq_range[0], freq_range[1]])
    
    # Set axis labels
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    
    # Load and plot labels
    try:
        with open(labeling_txt, 'r') as f:
            lines = f.readlines()
        
        # Define colors for different classes
        colors = ['red', 'blue', 'green']  # t3, t3b, t2
        class_names = ['t3', 't3b', 't2']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Skipping invalid line: {line}")
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized YOLO coordinates to time and frequency coordinates
            # x_center is normalized time position (0-1), convert to actual time
            time_center = time_range[0] + x_center * (time_range[1] - time_range[0])
            # y_center is normalized frequency position (0-1), convert to actual frequency
            freq_center = freq_range[0] + y_center * (freq_range[1] - freq_range[0])
            
            # Convert normalized width and height to actual time and frequency ranges
            time_width = width * (time_range[1] - time_range[0])
            freq_height = height * (freq_range[1] - freq_range[0])
            
            # Calculate rectangle corners in time-frequency coordinates
            time_min = time_center - time_width / 2
            freq_min = freq_center - freq_height / 2
            
            # Create rectangle patch in time-frequency coordinates
            rect = patches.Rectangle(
                (time_min, freq_min), time_width, freq_height,
                linewidth=2, edgecolor=colors[class_id % len(colors)],
                facecolor='none', alpha=0.8
            )
            
            # Add rectangle to plot
            plt.gca().add_patch(rect)
            
            # Add class label at the center of the bounding box
            class_name = class_names[class_id % len(class_names)]
            plt.text(time_center, freq_center, class_name, 
                    color=colors[class_id % len(colors)], 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        print(f"Plotted {len(lines)} labels from {labeling_txt}")
        
    except Exception as e:
        print(f"Error reading label file: {e}")
        return
    
    plt.title(f'Image: {os.path.basename(img_file)} | Labels: {os.path.basename(labeling_txt)}')
    plt.tight_layout()
    plt.show()


def save_config_to_yml(freq_range: List[float], t_res: float, t_start: float, 
                       N_freq: int, N_time: int, output_file: str = 'burst_config.yml') -> str:
    """Save burst generation configuration parameters to a YAML file.
    
    Args:
        freq_range (List[float]): Frequency range [min_freq, max_freq] in MHz.
        t_res (float): Time resolution in seconds.
        t_start (float): Start time in seconds.
        N_freq (int): Number of frequency channels.
        N_time (int): Number of time steps.
        output_file (str, optional): Output YAML file path. Defaults to 'burst_config.yml'.
    
    Returns:
        str: Path to the created configuration file.
    
    Example:
        >>> save_config_to_yml([30, 85], 0.5, 0.0, 640, 640)
        'burst_config.yml'
    """
    config = {
        'freq_range': freq_range,
        't_res': t_res,
        't_start': t_start,
        'N_freq': N_freq,
        'N_time': N_time
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return output_file


def load_config_from_yml(config_file: str = 'burst_config.yml') -> Dict[str, Any]:
    """Load burst generation configuration parameters from a YAML file.
    
    Args:
        config_file (str, optional): Path to the YAML configuration file. 
            Defaults to 'burst_config.yml'.
    
    Returns:
        Dict[str, Any]: Dictionary containing the configuration parameters.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    
    Example:
        >>> config = load_config_from_yml('burst_config.yml')
        >>> print(config['freq_range'])
        [30, 85]
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{config_file}': {e}")


def export_yolo_label(bursts: List[List[float]], burst_type: List[bool], 
                     output_dir: str = '.', base_filename: str = 'burst') -> str:
    """Export YOLO format labels for training object detection models.
    
    This function creates a YOLO label file (.txt) containing the bounding box annotations
    for solar radio bursts. The labels are in YOLO format: <class> <x_center> <y_center> <width> <height>
    where all coordinates are normalized to [0, 1].
    
    Args:
        bursts (List[List[float]]): List of bounding boxes in YOLO format 
            [x_center, y_center, width, height] for each burst.
        burst_type (List[bool]): List indicating whether each burst has fine structure (True for t3b, False for t3).
        output_dir (str, optional): Directory to save the label file. Defaults to '.'.
        base_filename (str, optional): Base name for the output file. Defaults to 'burst'.
    
    Returns:
        str: Path to the created label file.
    
    Note:
        - Type 0: t3 (Type III without fine structure), 1: t3b (Type III with fine structure)
        - The bounding boxes should already be in normalized YOLO format from the burst generation functions.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create label filename
    label_filename = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Open file and write YOLO labels
    with open(label_filename, 'w') as f:
        for i, (bbox, is_fine_structure) in enumerate(zip(bursts, burst_type)):
            # Determine class based on burst type
            # 0: t3 (Type III without fine structure), 1: t3b (Type III with fine structure)
            class_id = 1 if is_fine_structure else 0
            
            # Extract bounding box coordinates (already in YOLO format)
            x_center, y_center, width, height = bbox
            
            # Ensure coordinates are within [0, 1] range
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            width = np.clip(width, 0.0, 1.0)
            height = np.clip(height, 0.0, 1.0)
            
            # Write YOLO format line: <class> <x_center> <y_center> <width> <height>
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return label_filename


def mask_to_bbox(mask: np.ndarray, min_area: int = 10) -> List[List[float]]:
    """Convert a binary mask to YOLO format bounding boxes using largest connected components.
    
    This function finds the largest connected areas in a binary mask and converts them to
    YOLO format bounding boxes [center_x, center_y, width, height] with normalized coordinates.
    
    Args:
        mask (np.ndarray): Binary mask where True/1 indicates foreground pixels.
        min_area (int, optional): Minimum area threshold for connected components. Defaults to 10.
    
    Returns:
        List[List[float]]: List of bounding boxes in YOLO format [center_x, center_y, width, height].
    
    Example:
        >>> mask = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
        >>> bboxes = mask_to_bbox(mask)
        >>> print(bboxes)  # [[0.67, 0.33, 0.67, 0.67]]
    """
    import cv2
    

    # Check if mask has any foreground pixels before processing
    if np.sum(mask) == 0:
        return []  # Return empty list for empty mask

    # array and image are x-y swapped
    mask = mask.T
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8))

    # find the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Check if we have any components beyond background (label 0)
    if num_labels <= 1:
        return []  # Only background component exists
    
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask = (labels == largest_component).astype(np.uint8)

    # Ensure mask is binary and uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Since we already have the largest component mask, we can directly compute its bounding box
    # Find the bounding box of the non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return []  # No foreground pixels
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Convert to YOLO format (normalized coordinates)
    img_height, img_width = mask.shape
    center_x = (cmin + (cmax - cmin) / 2) / img_width
    center_y = (rmin + (rmax - rmin) / 2) / img_height
    width = (cmax - cmin + 1) / img_width
    height = (rmax - rmin + 1) / img_height
    
    # Ensure coordinates are within [0, 1]
    center_x = np.clip(center_x, 0.0, 1.0)
    center_y = np.clip(center_y, 0.0, 1.0)
    width = np.clip(width, 0.0, 1.0)
    height = np.clip(height, 0.0, 1.0)
    
    return [[center_x, center_y, width, height]]


def mask_to_all_bboxes(mask: np.ndarray, min_area: int = 10) -> List[List[float]]:
    """Convert a binary mask to all YOLO format bounding boxes for connected components.
    
    This function finds all connected areas in a binary mask above the minimum area threshold
    and converts them to YOLO format bounding boxes [center_x, center_y, width, height] 
    with normalized coordinates, sorted by area (largest first).
    
    Args:
        mask (np.ndarray): Binary mask where True/1 indicates foreground pixels.
        min_area (int, optional): Minimum area threshold for connected components. Defaults to 10.
    
    Returns:
        List[List[float]]: List of bounding boxes in YOLO format [center_x, center_y, width, height],
                           sorted by area (largest first).
    
    Example:
        >>> mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        >>> bboxes = mask_to_all_bboxes(mask)
        >>> print(bboxes)  # [[0.5, 0.5, 0.67, 0.67], [0.83, 0.83, 0.33, 0.33]]
    """
    import cv2
    

    # array and image are x-y swapped
    mask = mask.T
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8))

    # Ensure mask is binary and uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Check if we have any components beyond background (label 0)
    if num_labels <= 1:
        return []  # Only background component exists
    
    bboxes = []
    areas = []
    
    # Skip background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            # Get bounding box coordinates
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Convert to YOLO format (normalized coordinates)
            img_height, img_width = mask.shape
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            
            # Ensure coordinates are within [0, 1]
            center_x = np.clip(center_x, 0.0, 1.0)
            center_y = np.clip(center_y, 0.0, 1.0)
            width = np.clip(width, 0.0, 1.0)
            height = np.clip(height, 0.0, 1.0)
            
            bboxes.append([center_x, center_y, width, height])
            areas.append(area)
    
    # Sort by area (largest first)
    if bboxes:
        sorted_indices = np.argsort(areas)[::-1]
        bboxes = [bboxes[i] for i in sorted_indices]
    
    return bboxes


def mask_to_allpix_bbox(mask: np.ndarray, min_area: int = 10) -> List[float]:
    """Convert a binary mask to a single bounding box using all pixel coordinates.
    
    This function finds the bounding box that encompasses all True/1 pixels in the mask
    and returns it in YOLO format (normalized center x, center y, width, height).
    
    Args:
        mask (np.ndarray): Binary mask where True/1 indicates foreground pixels.
        
    Returns:
        List[float]: Bounding box in YOLO format [center_x, center_y, width, height].
        
    Example:
        >>> mask = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
        >>> bbox = mask_to_allpix_bbox(mask)
        >>> print(bbox)  # [0.5, 0.33, 0.67, 0.67]
    """
    # Find all pixel coordinates where mask is True
    x_indices, y_indices = np.where(mask)
    
    if len(y_indices) > 0 and len(x_indices) > 0:
        # Get absolute coordinates
        xmin = np.min(x_indices)
        ymin = np.min(y_indices)
        xmax = np.max(x_indices)
        ymax = np.max(y_indices)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        img_height, img_width = mask.shape
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        bbox = [x_center, y_center, width, height]
    else:
        bbox = [0.5, 0.5, 1.0, 1.0]  # default to full image if no mask found
    
    return [bbox]


def visualize_mask_and_bboxes(mask: np.ndarray, bboxes: List[List[float]], 
                             title: str = "Mask and Bounding Boxes") -> None:
    """Visualize a binary mask with overlaid bounding boxes.
    
    This function creates a plot showing the binary mask with bounding boxes overlaid
    for debugging and verification purposes.
    
    Args:
        mask (np.ndarray): Binary mask where True/1 indicates foreground pixels.
        bboxes (List[List[float]]): List of bounding boxes in YOLO format [center_x, center_y, width, height].
        title (str, optional): Title for the plot. Defaults to "Mask and Bounding Boxes".
    
    Example:
        >>> mask = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
        >>> bboxes = mask_to_bbox(mask)
        >>> visualize_mask_and_bboxes(mask, bboxes)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create figure
    plt.figure()
    
    # Plot mask
    # Use origin='upper' to match standard image coordinate system
    # where (0,0) is at top-left and y increases downward
    plt.imshow(mask, cmap='gray', origin='upper')
    
    # Plot bounding boxes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, bbox in enumerate(bboxes):
        center_x, center_y, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        # Note: mask.shape is (height, width) = (rows, columns)
        # YOLO format: [center_x, center_y, width, height] where:
        # - center_x is normalized column position (0-1)
        # - center_y is normalized row position (0-1)
        img_height, img_width = mask.shape
        x = center_x * img_width  # Column position
        y = center_y * img_height # Row position
        w = width * img_width     # Width in columns
        h = height * img_height   # Height in rows
        
        # Calculate rectangle corners
        # For origin='upper', y increases downward (standard image coordinates)
        x_min = x - w / 2
        y_min = y - h / 2
        
        # Create rectangle patch
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=color,
            facecolor='none', alpha=0.8
        )
        
        # Add rectangle to plot
        plt.gca().add_patch(rect)
        
        # Add box number
        plt.text(x, y, f'Box {i+1}', 
                color=color, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.colorbar(label='Mask Value')
    plt.tight_layout()
    plt.show()