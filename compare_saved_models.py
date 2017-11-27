# -*- coding: utf-8 -*-
"""
Compare weights of two models layer wise.
"""

from keras.models import load_model
import numpy as np
import argparse


def parse_input():
    parser = argparse.ArgumentParser(description='Compare weights of two models layer wise.')
    parser.add_argument('model_path_1', type=str, help='Model file 1 (.h5 file)')
    parser.add_argument('model_path_2', type=str, help='Model file 2 (.h5 file)')
    
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
model_path_1 = params["model_path_1"]
model_path_2 = params["model_path_2"]

#model_path_1 = "models/vgg_3_small/trained_vgg_3_small_autoencoder_epoch6.h5"
#model_path_2 = "models/vgg_3_small/trained_vgg_3_small_autoencoder_epoch6_supervised_up_down_epoch1.h5"


def do_model_check(model_path_1, model_path_2):
    model_1 = load_model(model_path_1)
    model_2 = load_model(model_path_2)
    
    for layer_no,layer_1 in enumerate(model_1.layers):
        layer_2 = model_2.layers[layer_no]
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

do_model_check(model_path_1, model_path_2)
