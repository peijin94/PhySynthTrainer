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
    """
    Generate a quasi-periodic signal for given time points.
    
    Parameters:
    -----------
    t_arr : array-like
        Array of time points
    base_freq : float
        Base frequency of the signal in Hz
    num_harmonics : int
        Number of harmonic components to include
    noise_level : float
        Amplitude of random noise (0 to 1)
    freqvar : float
        Amount of random frequency variation (0 to 1)
        
    Returns:
    --------
    array
        Signal values corresponding to input time points
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


def create_radio_burst_hash_table(freq_range=[30, 85], N_freq=640, v_range=[0.05, 0.5], N_v=200):
    """
    Generate lookup table for time offset of a type III radio burst
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
    return A0 * (np.exp(-(x-x0)**2/2/tau1**2) * (np.sign(x-x0)+1)/2 + np.exp(-(x-x0)**2/2/tau2**2) * (np.sign(x0-x)+1)/2)


def generate_type_iii_burst(freq_range=[30, 85], t_res=0.5, t_start=0.0, N_freq=640, N_time=640,
                           v_beam=0.15, t0_burst=80.0, decay_t_dur_100Mhz=1.0,
                           Burst_intensity=1.0, burststarting_freq=40.0,
                           burstending_freq=60.0, edge_freq_ratio=0.01,
                           fine_structure=True, use_hash_table=False, hash_table=None, v_hash=None):
    """
    Generate a Type III solar radio burst image.
    
    Parameters:
        freq_range (list): [min_freq, max_freq] in MHz
        t_res (float): Time resolution in seconds
        t_start (float): Start time in seconds
        N_freq (int): Number of frequency channels
        N_time (int): Number of time steps
        v_beam (float): Beam velocity in units of c
        t0_burst (float): Burst start time in seconds
        decay_t_dur_100Mhz (float): Decay time duration at 100 MHz
        Burst_intensity (float): Peak intensity of the burst
        burststarting_freq (float): Starting frequency in MHz
        burstending_freq (float): Ending frequency in MHz
        edge_freq (float): Edge smoothing frequency in MHz
        fine_structure (bool): Whether to add fine structure
        
    Returns:
        tuple: (img_bursts, t_ax, f_ax) - The burst image and its axes
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

    # make mask and bbox for training perposes
    mask = img_bursts > 0.1*np.max(img_bursts)
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


    return img_bursts, mask, bbox

import numpy as np
from typing import List, Tuple

def generate_many_random_t3_bursts(n_bursts: int = 100,
                         freq_range: List[float] = [30, 85],
                         t_res: float = 0.5,
                         t_start: float = 0.0,
                         N_freq: int = 640,
                         N_time: int = 640) -> List[Tuple]:
    """
    Generate multiple Type III radio bursts with random parameters.
    
    Parameters:
        n_bursts: Number of bursts to generate
        freq_range: [min_freq, max_freq] in MHz
        t_res: Time resolution in seconds
        t_start: Start time in seconds
        N_freq: Number of frequency channels
        N_time: Number of time steps
    
    Returns:
        List of tuples (img_bursts, mask, bbox) for each burst
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
        alpha = 2.5  # power law exponent, higher value means stronger events are rarer
        u = np.random.uniform(0, 1)
        Burst_intensity = 0.1 + 0.9 * ((1 - u) ** (-1/(alpha-1)))
        # Clip to ensure we stay in [0.1, 1] range
        Burst_intensity = np.clip(Burst_intensity, 0.02, 2.0)
        
        # Generate burststarting_freq and ensure burstending_freq is valid
        burststarting_freq = np.random.uniform(28, 60)
        min_ending = burststarting_freq + 4  # minimum ending frequency
        max_ending = min(88, burststarting_freq + 60)  # maximum ending frequency, capped at 100
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
            use_hash_table=True, hash_table=t_burst_hash, v_hash=v_ax
        )
        
        if np.max(img_bursts) > 0.001:    
            img_bursts_collect += img_bursts
            bursts.append((bbox))
            is_t3b.append(fine_structure)

    return img_bursts_collect, bursts, is_t3b

#img_bursts, bursts, is_t3b = generate_random_bursts(n_bursts=40)

