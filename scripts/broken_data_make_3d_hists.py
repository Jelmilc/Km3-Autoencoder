# -*- coding: utf-8 -*-
"""
This shows the impact of broken_data_mode in 3d plots of events.
They are taken from the generator, just like in training.
"""
import h5py
import numpy as np

from util.run_cnn import generate_batches_from_hdf5_file
from get_dataset_info import get_dataset_info
from plotting.make_autoencoder_3d_output_plot import make_3d_plots, reshape_3d_to_3d


which_broken_mode=1
batchsize=1
min_counts=0.1

titles=["Manipulated simulation", "Original simulation as 'measured' data"]
suptitle="Simulated data with added up-down information"

dataset_info_dict=get_dataset_info("debug_xzt")
is_autoencoder=True
class_type=None

filepath=dataset_info_dict["train_file"]
zero_center_image=dataset_info_dict["zero_center_image"]
n_bins = dataset_info_dict["n_bins"]

xs_mean=np.load(zero_center_image)

generator_normal = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=0, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=False, swap_col=None, is_in_test_mode = False)
data_normal=np.add(next(generator_normal)[0], xs_mean)
dataset_info_dict["broken_simulations_mode"]=which_broken_mode

generator_broken = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=which_broken_mode, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=False, swap_col=None, is_in_test_mode = False)
data_broken=np.add(next(generator_broken)[0], xs_mean)

for i in range(len(data_normal)):
    fig = make_3d_plots(reshape_3d_to_3d(data_broken[i], min_counts), reshape_3d_to_3d(data_normal[i], min_counts), n_bins[:-1], suptitle=suptitle, figsize=(12,7), titles=titles)

