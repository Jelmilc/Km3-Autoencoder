# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""
import argparse
import os

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training. Can also enter "saved" and a tag to restore saved plot properties.')
    parser.add_argument('autoencoder_model', type=str, help='name of _test.txt files to plot (or "saved")')
    parser.add_argument('parallel_model', type=str, help='name of _test.txt files to plot. (or a tag)')
    
    parser.add_argument('-c', '--channel', action="store_true", help='Use the channel parallel schedule (1 enc epoch per AE epoch)')
    parser.add_argument('-s', '--save', action="store_true", help='Save the plot to the path defined in the file.')
    args = parser.parse_args()
    params = vars(args)
    return params
params = parse_input()

import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y_parallel, get_last_prl_epochs
from scripts.util.saved_setups_for_plot_statistics import get_props_for_plot_parallel, get_how_many_epochs_each_to_train

test_files = [ params["autoencoder_model"], params["parallel_model"] ]
save_it = params["save"]

#Epoch schedule that was used during training as defined by the parser:
if params["channel"]==True:
    #used for channel AE
    epoch_schedule="1-1-1"
else:
    #used for everything else
    epoch_schedule="10-2-1"
    
#Default Values for plotting:
xlabel="Epoch"
<<<<<<< Updated upstream
title="Parallel Training"

#For putting 2 plots next to each other, this is alright and readable
style="two_in_one_line"

=======
title=""
figsize = (13,8)
>>>>>>> Stashed changes
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=["autoencoder", "supervised encoder training with dropout=0.1"]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=["blue", "orange"] # = automatic
#Name of file in the results/dumped_statistics folder to save the numpy array 
#with the plot data to; None will skip saving
dump_to_file=None
#save plot as
save_as=None


def make_parallel_statistics(test_files, title, labels_override, save_as, epoch_schedule, save_it, tag=None, style="two_in_one_line"):
    #Save and return the plot
    #Input: Either infos about what to plot, or a tag to load it automatically
    
    if tag != None:
        test_files, title, labels_override, save_as, epoch_schedule, style, ylims, AE_yticks = get_props_for_plot_parallel(tag)
        
    #Which epochs from the parallel encoder history to take:
    how_many_epochs_each_to_train = get_how_many_epochs_each_to_train(epoch_schedule)

    #Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list, default_label_array) 
    #for every test file
    data_from_files, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file=dump_to_file)
    
    data_autoencoder = data_from_files[0]
    data_parallel = data_from_files[1]
    
    data_parallel_test, data_parallel_train = get_last_prl_epochs(data_autoencoder, data_parallel, how_many_epochs_each_to_train)
    
    
    fig = make_plot_same_y_parallel(data_autoencoder, data_parallel_train, data_parallel_test, default_label_array, xlabel, ylabel_list, 
                     title, legend_locations, labels_override, colors, xticks, style, ylims=ylims, AE_yticks=AE_yticks)
    
    if save_as != None and save_it==True:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as)
        print("Saved plot as",save_as,"\n")
    else:
        print("Plot was not saved.")
    
    return fig

if test_files[0]=="saved":
    #overwrite some of the above options from a saved setup
    tag = test_files[1]
    
    if tag == "all":
        #make and save all the plots whose setup is saved, without displaying them.
        current_tag_number=0
        save_it=True
        while True:
            try:
                fig = make_parallel_statistics(test_files, title, labels_override, save_as, epoch_schedule, save_it, tag=current_tag_number, style=style)
                plt.close(fig)
                current_tag_number+=1
            except NameError:
                print("Done. Generated a total of",current_tag_number,"plots.")
                break
    else:
        #make, save and show a single plot whose setup is saved
        fig = make_parallel_statistics(test_files, title, labels_override, save_as, epoch_schedule, save_it, tag, style)
        plt.show(fig)

else:
    #Plot the files that were given to the parser directly
    tag = None
    fig = make_parallel_statistics(test_files, title, labels_override, save_as, epoch_schedule, save_it, tag, style)
    plt.show(fig)
