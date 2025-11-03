import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from physynthtrainer import densityModel, plasmaFreq, freqDrift
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from PIL import Image


def generate_quasi_periodic_signal(t_arr, base_freq=1.0, num_harmonics=5, 
                                 noise_level=0.1, freqvar=0.3):
    """Generate a quasi-periodic signal for given time points.

    Args:
        t_arr (array-like): Array of time points
        base_freq (float, optional): Base frequency of the signal in Hz. 
            Defaults to 1.0.
        num_harmonics (int, optional): Number of harmonic components to include. 
            Defaults to 5.
        noise_level (float, optional): Amplitude of random noise (0 to 1). 
            Defaults to 0.1.
        freqvar (float, optional): Amount of random frequency variation (0 to 1). 
            Defaults to 0.3.

    Returns:
        array: Signal values corresponding to input time points
    """
    # Convert input to numpy array
    t = np.array(t_arr)
    
    # Initialize signal
    signal = np.zeros_like(t)
    
    # Add harmonic components with frequency drift
    for i in range(num_harmonics):
        freq = base_freq * (i + 1) * (1 + freqvar * np.random.randn())
        # Generate random phase shift
        phase = 2 * np.pi * np.random.rand()
        # Add harmonic with decreasing amplitude
        amplitude = 1 / (i + 1)**0.5
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add random noise
    noise = noise_level * np.random.randn(len(t))
    signal += noise
    # Normalize signal
    signal = signal / np.max(np.abs(signal))
    return signal


def create_radio_burst_hash_table(freq_range=[30, 85], N_freq=640, v_range=[0.05, 0.5], N_v=200, t0_burst=80.0):
    """Generate lookup table for time offset of a type III radio burst.

    Args:
        freq_range (list, optional): [min_freq, max_freq] in MHz. 
            Defaults to [30, 85].
        N_freq (int, optional): Number of frequency channels. 
            Defaults to 640.
        v_range (list, optional): [min_v, max_v] in units of c. 
            Defaults to [0.05, 0.5].
        N_v (int, optional): Number of velocity steps. 
            Defaults to 200.
        t0_burst (float, optional): Burst start time in seconds. 
            Defaults to 80.0.

    Returns:
        tuple: (t_burst_collect, f_ax, v_ax) - The burst time offset table and its axes
    """
    f_ax = np.linspace(freq_range[0], freq_range[1], N_freq)
    v_ax = np.linspace(v_range[0], v_range[1], N_v)

    t_burst_collect = []
    for v_beam in np.linspace(v_range[0], v_range[1], N_v):
        t_burst = []
        for pix_f, f_this in enumerate(f_ax):
            t_burst.append(freqDrift.freq_drift_t_f(f_ax[pix_f], v_beam, t0_burst)[0])
        t_burst_collect.append(t_burst)

    t_burst_collect = np.array(t_burst_collect)
    return t_burst_collect, f_ax, v_ax


def biGaussian(x, x0, tau1, tau2, A0):
    """Calculate a bi-gaussian function.

    Args:
        x (array-like): Input array
        x0 (float): Center of the gaussian
        tau1 (float): Decay time for the first gaussian
        tau2 (float): Decay time for the second gaussian
        A0 (float): Amplitude of the gaussian

    Returns:
        array: The calculated bi-gaussian function
    """
    return A0 * (np.exp(-(x-x0)**2/2/tau1**2) * (np.sign(x-x0)+1)/2 + np.exp(-(x-x0)**2/2/tau2**2) * (np.sign(x0-x)+1)/2)


def generate_type_iii_burst(freq_range=[30, 85], t_res=0.5, t_start=0.0, N_freq=640, N_time=640,
                           v_beam=0.15, t0_burst=80.0, decay_t_dur_100Mhz=1.0,
                           Burst_intensity=1.0, burststarting_freq=40.0,
                           burstending_freq=60.0, edge_freq_ratio=0.01,
                           fine_structure=True, use_hash_table=False, hash_table=None, v_hash=None):
    """Generate a Type III solar radio burst image.

    Args:
        freq_range (list, optional): [min_freq, max_freq] in MHz. 
            Defaults to [30, 85].
        t_res (float, optional): Time resolution in seconds. 
            Defaults to 0.5.
        t_start (float, optional): Start time in seconds. 
            Defaults to 0.0.
        N_freq (int, optional): Number of frequency channels. 
            Defaults to 640.
        N_time (int, optional): Number of time steps. 
            Defaults to 640.
        v_beam (float, optional): Beam velocity in units of c. 
            Defaults to 0.15.
        t0_burst (float, optional): Burst start time in seconds. 
            Defaults to 80.0.
        decay_t_dur_100Mhz (float, optional): Decay time duration at 100 MHz. 
            Defaults to 1.0.
        Burst_intensity (float, optional): Peak intensity of the burst. 
            Defaults to 1.0.
        burststarting_freq (float, optional): Starting frequency in MHz. 
            Defaults to 40.0.
        burstending_freq (float, optional): Ending frequency in MHz. 
            Defaults to 60.0.
        edge_freq_ratio (float, optional): Edge smoothing frequency ratio. 
            Defaults to 0.01.
        fine_structure (bool, optional): Whether to add fine structure. 
            Defaults to True.
        use_hash_table (bool, optional): Whether to use a hash table for burst generation. 
            Defaults to False.
        hash_table (array, optional): The hash table to use for burst generation. 
            Defaults to None.
        v_hash (array, optional): The velocity hash to use for burst generation. 
            Defaults to None.

    Returns:
        tuple: (img_bursts, mask, bbox) - The burst image, its mask and bounding box
    """
    # Initialize arrays
    img_bursts = np.zeros((N_time, N_freq))
    t_ax = t_start + np.arange(0, N_time) * t_res
    f_ax = np.linspace(freq_range[0], freq_range[1], N_freq)
    
    t_burst = []
    
    edge_freq = (burstending_freq - burststarting_freq) * edge_freq_ratio
    # Generate fine structure if requested
    if fine_structure:
        freq_modulation = generate_quasi_periodic_signal(
            t_arr=f_ax, base_freq=0.5, num_harmonics=5,
            noise_level=0, freqvar=0.1
        )
        # normalize to 0-1
        freq_modulation = (freq_modulation - np.min(freq_modulation)) / (
            np.max(freq_modulation) - np.min(freq_modulation)
        )
    
    # Generate burst for each frequency
    for pix_f, f_this in enumerate(f_ax):
        if use_hash_table and hash_table is not None:
            # find closest velocity in hash table
            v_lookup = np.argmin(np.abs(v_hash - v_beam))
            t_burst.append(hash_table[v_lookup][pix_f]+ t0_burst)
        else:
            t_burst.append(freqDrift.freq_drift_t_f(f_ax[pix_f], v_beam, t0_burst))
        
        # generate burst time from a lookup table
        
        decay_t_dur_this = decay_t_dur_100Mhz * (100.0/f_this)
        
        # Calculate burst amplitude with frequency-dependent tapering
        burst_amp = 1 * (
            np.tanh((f_this - burststarting_freq) / edge_freq) + 1) * (
            np.tanh((burstending_freq - f_this) / edge_freq) + 1)
        
        # Apply fine structure modulation if requested
        if fine_structure:
            burst_amp = burst_amp * freq_modulation[pix_f]
        
        # Generate light curve for this frequency
        l_curve = biGaussian(t_ax, t_burst[-1], decay_t_dur_this, decay_t_dur_this/3, burst_amp)
        
        # Add to image array
        img_bursts[:, pix_f] += l_curve
    
    # normalize image to 0-1
    if np.max(img_bursts) > 1e-6:
        img_bursts = (img_bursts - np.min(img_bursts)) / (np.max(img_bursts) - np.min(img_bursts))
    else:
        img_bursts = img_bursts * 0.0
    img_bursts *= Burst_intensity

    # make mask and bbox for training purposes using the mask_to_bbox function
    mask = img_bursts > 0.03*np.max(img_bursts)
    
    # Use the mask_to_bbox function to get the largest connected component
    from .utils import mask_to_all_bboxes, mask_to_allpix_bbox
    bboxes = mask_to_allpix_bbox(mask, min_area=20)  # Minimum area threshold for type III bursts
    
    if bboxes:
        ## Take the largest connected component (first one after sorting by area)
        bbox = bboxes[0]
        
    else:
        bbox = [0.5, 0.5, 1.0, 1.0]  # default to full image if no mask found


    return img_bursts, mask, bbox

import numpy as np
from typing import List, Tuple


def generate_many_random_t3_bursts(n_bursts: int = 100,
                         freq_range: List[float] = [30, 85],
                         t_res: float = 0.5,
                         t_start: float = 0.0,
                         N_freq: int = 640,
                         N_time: int = 640,
                         use_hash_table: bool = False,
                         hash_table: np.ndarray = None,
                         v_hash: np.ndarray = None) -> List[Tuple]:
    """Generate multiple Type III radio bursts with random parameters.

    Args:
        n_bursts (int, optional): Number of bursts to generate. 
            Defaults to 100.
        freq_range (List[float], optional): [min_freq, max_freq] in MHz. 
            Defaults to [30, 85].
        t_res (float, optional): Time resolution in seconds. 
            Defaults to 0.5.
        t_start (float, optional): Start time in seconds. 
            Defaults to 0.0.
        N_freq (int, optional): Number of frequency channels. 
            Defaults to 640.
        N_time (int, optional): Number of time steps. 
            Defaults to 640.
        use_hash_table (bool, optional): Whether to use a hash table for burst generation. 
            Defaults to False.
        hash_table (np.ndarray, optional): The hash table to use for burst generation. 
            Defaults to None.
        v_hash (np.ndarray, optional): The velocity hash to use for burst generation. 
            Defaults to None.

    Returns:
        List[Tuple]: A list of tuples containing the generated bursts, their bounding boxes, and whether they are type 3b bursts
    """
    bursts = []
    is_t3b = []
    img_bursts_collect = np.zeros((N_time, N_freq))
    for _ in range(n_bursts):
        # Generate random parameters within specified ranges
        v_beam = np.random.uniform(0.08, 0.4)
        t0_burst = np.random.uniform(-80, 320)
        decay_t_dur_100Mhz = np.random.uniform(0.4, 1.5)


        # Using inverse transform sampling: x = (1 - u)^(-1/(alpha-1)) where u is uniform random
        alpha = 1.7  # power law exponent, higher value means stronger events are rarer
        u = np.random.uniform(0, 1)
        Burst_intensity = (0.1 + 0.9 * ((1 - u) ** (-1/(alpha-1)))) / 50 +0.02
        # Clip to ensure we stay in [0.1, 1] range
        
        # Generate burststarting_freq and ensure burstending_freq is valid
        burststarting_freq = np.random.uniform(freq_range[0], freq_range[1]-30)
        min_ending = burststarting_freq + 6  # minimum ending frequency
        max_ending = min(freq_range[1], burststarting_freq + 60)  # maximum ending frequency, capped at 100
        burstending_freq = np.random.uniform(min_ending, max_ending)
        
        edge_freq_ratio = np.random.uniform(0.05, 0.15)
        fine_structure = np.random.random() < 0.3   # Randomly True or False
        
        # Generate burst with random parameters
        img_bursts, mask, bbox = generate_type_iii_burst(
            freq_range=freq_range,
            t_res=t_res,
            t_start=t_start,
            N_freq=N_freq,
            N_time=N_time,
            v_beam=v_beam,
            t0_burst=t0_burst,
            decay_t_dur_100Mhz=decay_t_dur_100Mhz,
            Burst_intensity=Burst_intensity,
            burststarting_freq=burststarting_freq,
            burstending_freq=burstending_freq,
            edge_freq_ratio=edge_freq_ratio,
            fine_structure=fine_structure,
            use_hash_table=use_hash_table,
            hash_table=hash_table,
            v_hash=v_hash
        )
        
        if np.max(img_bursts) > 0.001:    
            img_bursts_collect += img_bursts
            bursts.append((bbox))
            is_t3b.append(fine_structure)

    return img_bursts_collect, bursts, is_t3b

#img_bursts, bursts, is_t3b = generate_many_random_t3_bursts(n_bursts=40)


def generate_type_2_burst(freq_range=[30, 85], t_res=0.5, t_start=0.0, N_freq=640, N_time=640,
                         v_shock=700, t_s_start=80.0, t_s_end=180.0, eff_starting_freq=100,
                         t0_R_start=0, laneNUM=6, harmonic_overlap=False):
    """Generate a type II radio burst with multiple harmonic lanes.
    
    Args:
        freq_range (list, optional): [min_freq, max_freq] in MHz. Defaults to [30, 85].
        t_res (float, optional): Time resolution in seconds. Defaults to 0.5.
        t_start (float, optional): Start time in seconds. Defaults to 0.0.
        N_freq (int, optional): Number of frequency channels. Defaults to 640.
        N_time (int, optional): Number of time steps. Defaults to 640.
        v_shock (float, optional): Shock velocity in km/s. Defaults to 700.
        t_s_start (float, optional): Shock start time in seconds. Defaults to 80.0.
        t_s_end (float, optional): Shock end time in seconds. Defaults to 180.0.
        eff_starting_freq (float, optional): Effective starting frequency in MHz. Defaults to 100.
        t0_R_start (float, optional): Time offset for radius calculation in seconds. Defaults to 0.
        laneNUM (int, optional): Number of harmonic lanes. Defaults to 6.
        harmonic_overlap (bool, optional): Whether to include second harmonic. Defaults to False.
    
    Returns:
        tuple: (img_bursts, mask, bbox) - The generated burst image, mask, and bounding box
    """
    # Initialize arrays
    img_bursts = np.zeros((N_time, N_freq))
    t_ax = t_start + np.arange(0, N_time) * t_res
    f_ax = np.linspace(freq_range[0], freq_range[1], N_freq)
    
    # Find time indices for shock start and end
    t_s_start_idx = np.argmin(np.abs(t_ax - t_s_start))
    t_s_end_idx = np.argmin(np.abs(t_ax - t_s_end))
    
    # Calculate radius and frequency evolution
    R_start = freqDrift.freq_to_R(eff_starting_freq * 1e6)
    R_all_t = R_start + v_shock * 1e3 * (t_ax + t0_R_start) / 2.99792458e8
    f_all_t = freqDrift.R_to_freq(R_all_t)
    
    # Fixed bandwidth ratio
    bandw = 0.1  # ratio of df/f
    bandw_freq_depend = bandw * (f_all_t / 1e6) ** 1
    
    # Random parameters for lanes
    smooth_edge = np.random.uniform(2, 10)
    f_ratio_upper_lim = np.random.uniform(1.2, 1.8)
    f_ratio_lane = np.random.uniform(1.0, f_ratio_upper_lim, laneNUM)
    f_ratio_lane[0] = 1.0  # First lane is fundamental
    bandw_radio = np.random.uniform(0.05, 0.8, laneNUM)
    amp_lane = np.random.uniform(0.3, 1.0, laneNUM)
    
    # Generate each harmonic lane
    for lane_idx in range(laneNUM):
        # Generate amplitude modulation in time
        amp_tx = generate_quasi_periodic_signal(
            t_arr=t_ax, base_freq=0.05, num_harmonics=5, 
            noise_level=0, freqvar=0.1
        )
        # Normalize to 0.1-1
        amp_tx = 0.1 + (amp_tx - np.min(amp_tx)) / (np.max(amp_tx) - np.min(amp_tx)) * 0.9
        # Apply smooth edges
        amp_tx = np.tanh((t_ax - t_s_start) / smooth_edge) * np.tanh((t_s_end - t_ax) / smooth_edge) * amp_tx
        
        # Scale by lane amplitude
        amp_tx = amp_tx * amp_lane[lane_idx]
        
        # Add lane to image
        for t_idx_event, this_t in enumerate(range(t_s_start_idx, t_s_end_idx)):
            f_this = f_all_t[this_t] / 1e6
            
            # Add fundamental frequency lane
            img_bursts[this_t, :] += amp_tx[this_t] * np.exp(
                -(f_ax - f_this * f_ratio_lane[lane_idx]) ** 2 / 
                (2 * bandw_radio[lane_idx] * (bandw_freq_depend[t_idx_event] / 2) ** 2)
            )
            
            # Add second harmonic if requested
            if harmonic_overlap:
                img_bursts[this_t, :] += amp_tx[this_t] * np.exp(
                    -(f_ax - f_this * f_ratio_lane[lane_idx] * 2) ** 2 / 
                    (2 * bandw_radio[lane_idx] * (bandw_freq_depend[t_idx_event] / 2) ** 2)
                )
    
    # Create mask and bounding box using the mask_to_bbox function
    threshold = np.max(img_bursts) * 0.03
    mask = img_bursts > threshold
    
    # Use the mask_to_bbox function to get the largest connected component
    from .utils import mask_to_all_bboxes, mask_to_bbox
    bboxes = mask_to_all_bboxes(mask, min_area=30)  # Minimum area threshold for type II bursts
    
    if bboxes:
        # Take the largest connected component (first one after sorting by area)
        bbox = bboxes
    else:
        # Default bbox if no mask or no components meet the area threshold
        bbox = [0.5, 0.5, 0.1, 0.1]
    
    return img_bursts, mask, bbox


def added_noise(t_ax, f_ax, noise_level=0.2, noise_size=[32,8]):
    """Add radio background noise to the image.

    Args:
        t_ax (array-like): Time axis of the image
        f_ax (array-like): Frequency axis of the image
        noise_level (float, optional): The level of noise to add. 
            Defaults to 0.2.
        noise_size (list, optional): The size of the noise to generate. 
            Defaults to [32,8].

    Returns:
        array: The interpolated noise
    """

    original = np.random.uniform(0.1,0.3, size=noise_size)
    x = np.linspace(0, 1, noise_size[0])
    y = np.linspace(0, 1, noise_size[1])

    x_N = t_ax.shape[0]
    y_N = f_ax.shape[0]
    x_new = np.linspace(0, 1, x_N)
    y_new = np.linspace(0, 1, y_N)

    f = RectBivariateSpline(x, y, original)
    interpolated = f(x_new, y_new)
    # normalize to 0-0.05
    interpolated_norm = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated)) * noise_level

    return interpolated_norm




