# -*- coding: utf-8 -*-

"""
Evaluate model performance after training for up-down or AUtoencoder networks.
Will save and display a binned histogram plot of acc or loss vs energy.

Specify trained models, either AE or supervised up-down, and calculate their loss or acc
on test data.
This is then binned and automatically dumped. Instead of recalculating, 
it is loaded automatically.
Can also plot it and save it to results/plots
"""

import matplotlib.pyplot as plt

from util.evaluation_utilities import make_or_load_files, make_binned_data_plot
from util.saved_setups_for_plot_statistics import get_path_best_epoch


tag="dpg_plot"

def get_saved_plots_info(tag):
    class_type = (2, 'up_down')
    #Type of plot which is generated for whole array (it should all be of the same type):
    #loss, acc, None
    plot_type = "acc"
    bins=32
    y_lims=(0.75,1) #for acc only
    
    full_path=False
    if tag=="dpg_plot":
        #Model info:
        #list of modelidents to work on (has to be an array, so add , at the end
        #if only one file)
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_3", full_path),
                       get_path_best_epoch("vgg_3_unf", full_path))
        #Dataset to evaluate on
        dataset_array = ["xzt",] * len(modelidents)
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Up-down Performance comparison'
        label_array=["Best from autoencoder 200", "Best from autoencoder 2000", "Best supervised"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "dpg_vgg3_vgg5_200_spvsd_comp.pdf" 
    
    elif tag=="compare_600":
        #morefilter and picture (evtl channel)
        modelidents = (get_path_best_epoch("vgg_5_600_picture", full_path),
                       get_path_best_epoch("vgg_5_600_morefilter", full_path),)
        
    elif tag=="compare_bottleneck":
        #morefilter and picture (evtl channel)
        modelidents = (get_path_best_epoch("vgg_3", full_path),
                       get_path_best_epoch("vgg_5_600_picture", full_path),
                       get_path_best_epoch("vgg_5_200", full_path),)
                       #get_path_best_epoch("vgg_5_64", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of autoencoders with different bottleneck sizes'
        label_array=["2000", "600", "200", "64"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_bottleneck_compare.pdf"
     
        
       

    else: raise NameError
    
    modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/updown_evaluation/"
    
    modelidents=[modelpath+modelident for modelident in modelidents]
    save_plot_as = plot_path + plot_file_name

    return modelidents, class_type, dataset_array, title_of_plot, label_array, save_plot_as, plot_type, bins, y_lims



def make_evaluation(tag):
    modelidents, class_type, dataset_array, title_of_plot, label_array, save_plot_as, plot_type, bins, y_lims = get_saved_plots_info(tag)
    
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

make_evaluation(tag)



