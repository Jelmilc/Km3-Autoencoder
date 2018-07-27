# -*- coding: utf-8 -*-

import argparse


"""
Make a plot of multiple models, each identified with its test log file, with
lots of different options.
This is intended for plotting models with the same y axis data (loss OR acc).
"""

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training. Can also enter "saved" and a tag to restore saved plot properties.')
    parser.add_argument('models', type=str, nargs="+", help='name of _test.txt files to plot. (or "saved" and a tag)')

    parser.add_argument('-s', '--save', action="store_true", help='Save the plot to the path defined in the file.')
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = params["models"]
save_it = params["save"]


import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
from scripts.util.saved_setups_for_plot_statistics import get_props_for_plot_parser

#Default Values:
xlabel="Epoch"

#Different titles for plots
title = ""
#title = "supervised parallel training"
#title="Loss of autoencoders with a varying number of convolutional layers"
figsize = (13,8)
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=[ "Encodertraining vgg-5-2000-eps01 on manipulated data", "Fully supervised training on manipulated data", "Fully supervised training on correct data"]
#legend location for the labels and the test/train box,
>>>>>>> Stashed changes
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None

# override line colors; must be one color for every test file, otherwise automatic
colors=[] # = automatic
#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None
#How hte plotting window should look like
style="extended"
#Save the plot, None to skip
save_as=None
#Ranges for the plot
xrange="auto"
#Average over this many bins in the train data (to reduce jitter)
average_train_data_bins=1


def make_parser_plot(test_files, title, labels_override, save_as, 
                     legend_locations, colors, xticks, style,
                     dump_to_file, xlabel, save_it, show_it, xrange, 
                     average_train_data_bins=1):
    #Read the data in, auto means take acc when available, otherwise loss
    data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, which_ydata="auto",
                                                                            dump_to_file=dump_to_file)
    #Create the plot
    fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                    legend_locations, labels_override, colors, xticks, style=style, 
                    xrange=xrange, average_train_data_bins=average_train_data_bins)
    #Save the plot
    if save_as != None and save_it==True:
        plt.savefig(save_as)
        print("Saved plot as",save_as)
    else:
        print("Plot was not saved.")
    
    if show_it: plt.show(fig)


data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file=dump_to_file)
fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title,
                legend_locations, labels_override, colors, xticks, figsize=figsize)

if save_as != None:
    plt.savefig(save_as)
    print("Saved plot as",save_as)

plt.show(fig)


