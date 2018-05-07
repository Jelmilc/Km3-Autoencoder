# -*- coding: utf-8 -*-

"""
Evaluate model performance after training
"""

import numpy as np
import matplotlib.pyplot as plt

from util.evaluation_utilities import make_or_load_files, make_binned_data_plot


"""
Specify trained models, either AE or supervised, and calculate their loss or acc on test data.
This data is then binned and automatically dumped. Instead of recalculating, 
it is loaded automatically.
Can also plot it and save it to results/plots
"""


#Model info:
#list of modelidents to work on (has to be an array, so add , at the end if only one file)
modelidents = ("vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_up_down_epoch45.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch23.h5",
               "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5")

class_type = (2, 'up_down')

#Dataset to evaluate on
dataset_array = ["xzt",] * len(modelidents)



#Plot properties: All in the array are plotted in one figure, with own label each
title_of_plot='Up-down Performance comparison'
label_array=["Best from autoencoder 200", "Best from autoencoder 2000", "Best supervised"]
plot_file_name = "dpg_vgg3_vgg5_200_spvsd_comp.pdf" #in the results/plots folder
#Type of plot which is generated for whole array (it should all be of the same type):
#loss, acc, None
plot_type = "acc"

bins=32
y_lims=(0.75,1) #for acc only




#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"

modelidents=[modelpath+modelident for modelident in modelidents]

save_plot_as = plot_path + plot_file_name



#generate or load data automatically:
hist_data_array = make_or_load_files(modelidents, dataset_array, bins, class_type)


#make plot of multiple data:
if plot_type == "acc":
    y_label_of_plot="Accuracy"
    fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot, y_lims=y_lims) 
    plt.show(fig)
    fig.savefig(save_plot_as)
    
elif plot_type == "loss":
    y_label_of_plot="Loss"
    fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot) 
    plt.show(fig)
    fig.savefig(save_plot_as)
    
elif plot_type == None:
    print("plot_type==None: Not generating plots")
else:
    print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")

print("Plot saved to", save_plot_as)





