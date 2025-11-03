import os
import gc

from physynthtrainer.burstGen import generate_many_random_t3_bursts, added_noise
from physynthtrainer import utils
import numpy as np

gc.collect()

dir_labels = './dataset/labels/'
dir_images = './dataset/images/'

# create directories if not exist
os.makedirs(dir_labels, exist_ok=True)
os.makedirs(dir_images, exist_ok=True)


totalset = 6000

for i in range(totalset):

    fname = f'b{i:05d}'

    num_bursts = np.random.randint(5, 60)

    img_bursts, bursts,t3b = generate_many_random_t3_bursts(n_bursts=num_bursts)
    interpolated_norm = added_noise(t_ax, f_ax, 0.2, noise_size=[32,8])
    y = np.linspace(0.1, 1.2, 640)
    const_bg = np.tile(y[:, np.newaxis], (1, 640))
    img_bursts_withbg = img_bursts+interpolated_norm.T+const_bg.T
    #plt.imshow(img_bursts_withbg.T, interpolation='nearest', aspect='auto', origin='lower')
    #for bbox in bursts:
    #    x_center, y_center, width, height = bbox
    #    xmin = int((x_center - width / 2) * img_width)
    #    xmax = int((x_center + width / 2) * img_width)
    #    ymin = int((y_center - height / 2) * img_height)
    #    ymax = int((y_center + height / 2) * img_height)
    #    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
    with open(dir_labels + fname + '.txt', 'w') as f:
        for is_t3b_this, bbox in zip(t3b, bursts):
            if is_t3b_this:
                f.write('1 ')
            else:
                f.write('0 ')
            x_center, y_center, width, height = bbox
            f.write(f'{x_center} {y_center} {width} {height}\n')

    utils.paint_arr_to_jpg(img_bursts_withbg, dir_images + fname + '.jpg', flip_y=True)