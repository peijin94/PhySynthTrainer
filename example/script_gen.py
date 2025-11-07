import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
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

from skimage.transform import resize
import gc, os
from multiprocessing import Pool, cpu_count

gc.collect()

dir_labels = '/data07/peijinz/ML/type3MLgen/dataset_v3/labels/'
dir_images = '/data07/peijinz/ML/type3MLgen/dataset_v3/images/'

# create directories if not exist
os.makedirs(dir_labels, exist_ok=True)
os.makedirs(dir_images, exist_ok=True)


totalset = 50000


t_burst_hash, f_ax, v_ax = create_radio_burst_hash_table(
    freq_range=[15, 85], 
    N_freq=640, 
    v_range=[0.04, 0.5], 
    N_v=200)


bgfile = '/data07/peijinz/ML/type3MLgen/PhySynthTrainer/physynthtrainer/data/latest_data.3.npz'


bg_lst_npz  = "/data07/peijinz/ML/type3MLgen/PhySynthTrainer/data/bckgrd_list.npz"

bgdata = np.load(bgfile)

data_bg = np.load(bg_lst_npz)
noise_bg = data_bg['bckgrd_list']


def process_single_burst(args):
    """Process a single burst generation iteration"""
    i, t_burst_hash, v_ax, noise_bg = args
    
    # Set random seed for this worker to ensure different random values
    np.random.seed(None)
    
    fname = f'b{i:05d}'
    
    num_bursts = np.random.randint(3, 40)
    
    max_norm_factor = np.random.uniform(2, 145)
    
    
    img_bursts, bursts, is_t3b = generate_many_random_t3_bursts(
        n_bursts=num_bursts, 
        use_hash_table=True, 
        freq_range=[22, 75],
        hash_table=t_burst_hash, 
        v_hash=v_ax,
        t_res=0.5,
        t_start=0.0
    )
    
    img_bursts = (img_bursts)/np.max(img_bursts) * max_norm_factor+0.2 # 0.2 is the background level
    
    idx_bg_rand = np.random.randint(0, noise_bg.shape[0])
    noise_bg_arr = resize(np.tile(noise_bg[idx_bg_rand], (10, 1)), img_bursts.shape, order=1)
    img_bursts_withbg = img_bursts+noise_bg_arr
    
    
    label_file = export_yolo_label(
        bursts, 
        is_t3b, 
        output_dir=dir_labels, 
        base_filename=fname)
    
    paint_arr_to_jpg(img_bursts_withbg, dir_images + fname + '.jpg', flip_y=False, scaling='log'
        ,vmax=150,vmin=0.5)
    
    return i


if __name__ == '__main__':
    # Save config file first
    config_file = save_config_to_yml(
        freq_range=[15, 85],
        t_res=0.5,
        t_start=0.0,
        N_freq=640,
        N_time=640,
        output_file='base.yml'
    )
    
    # Prepare arguments for multiprocessing
    args_list = [(i, t_burst_hash, v_ax, noise_bg) for i in range(totalset)]
    
    # Use multiprocessing with all available CPU cores
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores for parallel processing...")
    
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance with progress bar
        list(tqdm(pool.imap_unordered(process_single_burst, args_list), 
                  total=totalset, 
                  desc="Generating bursts"))