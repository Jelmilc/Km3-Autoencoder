# -*- coding: utf-8 -*-
"""
Compare weights of two models layer wise.
"""
<<<<<<< Updated upstream

from keras.models import load_model, Model
=======
from keras.models import load_model
>>>>>>> Stashed changes
import numpy as np
import argparse

from scripts.util.custom_loss_functions import get_custom_objects


def parse_input():
    parser = argparse.ArgumentParser(description='Compare weights of two models layer wise.')
    parser.add_argument('model_path_1', type=str, help='Model file 1 (.h5 file)')
    parser.add_argument('model_path_2', type=str, help='Model file 2 (.h5 file)')
    
    args = parser.parse_args()
    params = vars(args)
    return params


param = parse_input()
model_path_1 = param["model_path_1"]
model_path_2 = param["model_path_2"]

#model_path_1 = "models/vgg_3_small/trained_vgg_3_small_autoencoder_epoch6.h5"
#model_path_2 = "models/vgg_3_small/trained_vgg_3_small_autoencoder_epoch6_supervised_up_down_epoch1.h5"

def print_layer_stats(layer_1, layer_2):
    print("Layer Names:")
    print("\t", layer_1.name, "\t", layer_2.name)
    weights_1 = np.array(layer_1.get_weights())
    weights_2 = np.array(layer_2.get_weights())
    print("Shape of weights matrix:")
    for i,filt_1 in enumerate(weights_1):
        filt_2=weights_2[i]
        print("\t", filt_1.shape, "\t", filt_2.shape)  
        
        fl_w_1 = filt_1.flatten()
        fl_w_2 = filt_2.flatten()
        
        print("\tEqual? ",np.array_equal(fl_w_1, fl_w_2))
    print("\n")

def do_model_check(model_path_1, model_path_2):
    custom_objects=get_custom_objects()
    model_1 = load_model(model_path_1, custom_objects=custom_objects)
    model_2 = load_model(model_path_2, custom_objects=custom_objects)
    
    for layer_no, layer_1 in enumerate(model_1.layers):
        layer_2 = model_2.layers[layer_no]
<<<<<<< Updated upstream
        
        if not isinstance(layer_1, Model):
            print_layer_stats(layer_1, layer_2)
            
        else:
            #layer 1 is a model
            for sub_num, layer in enumerate(layer_1.layers):
                layer_2_sub = model_2.layers[sub_num]
                print_layer_stats(layer, layer_2_sub)
            break
=======
        print("Layer Names:")
        print("\t", layer_1.name, "\t", layer_2.name)
        weights_1 = np.array(layer_1.get_weights())
        weights_2 = np.array(layer_2.get_weights())
        print("Shape of weights matrix:")
        for i, filt_1 in enumerate(weights_1):
            filt_2 = weights_2[i]
            print("\t", filt_1.shape, "\t", filt_2.shape)  
            
            fl_w_1 = filt_1.flatten()
            fl_w_2 = filt_2.flatten()
            
            print("\tEqual? ", np.array_equal(fl_w_1, fl_w_2))
>>>>>>> Stashed changes
            
       


do_model_check(model_path_1, model_path_2)
