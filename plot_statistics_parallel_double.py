# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""
#TODO display train loss of parallel where possible (when schedule is at 1 AE E/1 par E)
import argparse
import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y_parallel, get_last_prl_epochs

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training.')
    parser.add_argument('autoencoder_model', type=str, help='name of _test.txt files to plot.')
    parser.add_argument('parallel_model', type=str, help='name of _test.txt files to plot.')
    parser.add_argument('parallel_model_2', type=str, help='name of _test.txt files to plot.')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = [ params["autoencoder_model"], params["parallel_model"], params["parallel_model_2"]]


xlabel="Epoch"
title=""
#title="Parallel Training with additional convolution"
figsize = (13,8)
#Override default labels (names of the models); must be one for every test file, otherwise default:Two dense layers,Convolution and two dense layers
labels_override=["Autoencoder", "encoder training(Dropout=0.2 in dense)", "encoder training(Dropout=0.1 in dense)"]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=["blue", "orange", "green"] # = automatic
#Name of file in the results/dumped_statistics folder to save the numpy array 
#with the plot data to; None will skip saving
dump_to_file=None



#Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list, default_label_array) 
#for every test file
data_from_files, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file)

data_autoencoder = data_from_files[0]
data_parallel = data_from_files[1]
data_parallel_2 = data_from_files[2]


#Which epochs from the parallel encoder history to take:
how_many_epochs_each_to_train = np.array([10, ]*1+[2, ]*5+[1, ]*200)

data_parallel_test, data_parallel_train = get_last_prl_epochs(data_autoencoder, data_parallel,
                                            how_many_epochs_each_to_train)
data_parallel_test_2, data_parallel_train_2 = get_last_prl_epochs(data_autoencoder, data_parallel_2,
                                            how_many_epochs_each_to_train)


data_parallel_2 = [data_parallel_test_2[0], data_parallel_test_2[1], data_parallel_train_2[0], data_parallel_train_2[1]]


fig = make_plot_same_y_parallel(data_autoencoder, data_parallel_train, data_parallel_test, default_label_array, xlabel, ylabel_list, 
                 title, legend_locations, labels_override, colors, xticks, figsize, data_parallel_2)
plt.show(fig)
