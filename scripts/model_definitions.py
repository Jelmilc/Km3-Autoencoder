# -*- coding: utf-8 -*-

"""
Functions that return models are defined in this file
"""

#from model_def.model_def_vgg_1 import setup_vgg_1, setup_vgg_1_xzt, setup_vgg_1_xzt_max, setup_vgg_1_xzt_stride
#from model_def.model_def_vgg_2_xzt import setup_vgg_2,setup_vgg_2_dropout, setup_vgg_2_max, setup_vgg_2_stride
from model_def.model_def_vgg_3 import setup_vgg_3,setup_vgg_3_dropout, setup_vgg_3_max, setup_vgg_3_stride, setup_vgg_3_stride_noRelu, setup_vgg_3_small, setup_vgg_3_verysmall, setup_vgg_3_reg
from model_def.model_def_vgg_4 import *
from model_def.model_def_vgg_5 import *
from model_def.model_def_vgg_6 import *
from model_def.model_def_channel_1 import *
import warnings

def make_options_dict(additional_options):
    #Can alter things like dropout rate or BN unlock etc.
    #additional_options is the string handed to the parser in run_autoencoder
    #via options
    
    #Default entries:
    options_dict={}
    options_dict["dropout_for_dense"] = 0.0
    options_dict["batchnorm_before_dense"]=True
    options_dict["unlock_BN_in_encoder"]=False
    options_dict["batchnorm_for_dense"]=False
    options_dict["encoder_only"]=False
    options_dict["encoded_penalty"]=0
    options_dict["dropout_for_conv"]=0.0
    options_dict["add_conv_layer"]=False
    options_dict["make_stateful"]=False
    options_dict["dense_setup"]="standard"
    options_dict["layer_version"]="single"
    options_dict["pretrained_autoencoder_path"]=None
    options_dict["pretrained_critic_path"]=None
        
    additional_options_list = additional_options.split("-")
    joined_list = []
    quotation_started=False
    join_sign="'"
    floating_option=""
    for i in range(len(additional_options_list)):
        additional_options = additional_options_list[i]
        #could also be sth like unlock_BN=true-load_model="asd-mas-d.h5"-bla=bla
        #would get split to :
        #   unlock_BN=true
        #   load_model="asd
        #   mas
        #   md.h5"
        #   bla=bla
        #--> merge entry 2-4 from the above!
        if join_sign in additional_options:
            additional_options = additional_options.replace(join_sign, '')
            if not quotation_started:
                quotation_started=True
                floating_option = additional_options + "-"
            else:
                floating_option = floating_option + additional_options
                quotation_started=False
                joined_list.append(floating_option)
            continue
            
        if quotation_started:
            floating_option = floating_option + additional_options + "-"
        else:
            joined_list.append(additional_options)
            
    if quotation_started:
        warnings.warn("Additional options contained only one", join_sign,"but an even number is required!")
        
        
    for additional_options in joined_list:
        if additional_options == "unlock_BN":
            #Always unlock the BN layers in the encoder part.
            options_dict["unlock_BN_in_encoder"]=True
            print("Batchnorms will be unlocked")
            
        elif "dropout=" in additional_options:
            #Change dropout of dense layers
            dropout_rate = float(additional_options.split("=")[1])
            options_dict["dropout_for_dense"]=dropout_rate
            print("Dropout rate for dense layers will be", dropout_rate)
        
        elif "dropout_conv=" in additional_options:
            #Change dropout of conv layers
            dropout_rate = float(additional_options.split("=")[1])
            options_dict["dropout_for_conv"]=dropout_rate
            print("Dropout rate for conv layers will be", dropout_rate)
            print("Warning: not supported for all models!")
        
        elif "l1reg=" in additional_options:
            #Apply l1 reguarizer to encoded layer
            penalty = float(additional_options.split("=")[1])
            options_dict["encoded_penalty"]=penalty
            print("L1 regularizer of encoded layer with factor", penalty)
        
        elif "add_conv_layer" in additional_options:
            #Add a trainable conv layer before the dense layers of the encoder
            options_dict["add_conv_layer"]=True
            print("Conv layer is being added to encoder")
            
        elif "encoder_only" in additional_options:
            #Change dropout of dense layers
            options_dict["encoder_only"]=True
            print("Encoder only mode enabled! Not compatible with all models (you might get an error)")
        
        elif "dense_setup=" in additional_options:
            #change the encoder part, only supported for vgg5picture
            #possible setups: standard, deep, shallow
            dense_setup = additional_options.split("=")[1]
            options_dict["dense_setup"]=dense_setup
            print("Using dense setup:",dense_setup, "(only supported for vgg5picture, vgg5_32)")
        
        elif "layer_version=" in additional_options:
            #Defines if the new (double c layer before 1st pooling) or the old 
            #version of the 64 and 32 networks should be used
            #either single or double
            layer_version = additional_options.split("=")[1]
            options_dict["layer_version"]=layer_version
            
        #For GAN AEs:
        elif "pretrained_autoencoder_path" in additional_options:
            pretrained_autoencoder_path = additional_options.split("=")[1]
            options_dict["pretrained_autoencoder_path"]=pretrained_autoencoder_path
        elif "pretrained_critic_path" in additional_options:
            pretrained_critic_path = additional_options.split("=")[1]
            options_dict["pretrained_critic_path"]=pretrained_critic_path

        else:
           warnings.warn("Ignoring unrecognized string for options:"+additional_options)
            
    return options_dict


def setup_model(model_tag, autoencoder_stage, modelpath_and_name=None, additional_options="", number_of_output_neurons=2):
    #model tags can have version numbers, e.g. vgg_1-different_lr
    #for this, load model without the version number
    splitted_tag = model_tag.split("-")
    model_tag = splitted_tag[0]
    if len(splitted_tag)==2:
        print("Creating model "+model_tag+" in version "+splitted_tag[1])
    
    options_dict = make_options_dict(additional_options)
    options_dict["number_of_output_neurons"]=number_of_output_neurons
    
    if model_tag == "vgg_1":
        model = setup_vgg_1(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt":
        model = setup_vgg_1_xzt(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt_max":
        model = setup_vgg_1_xzt_max(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt_stride":
        model = setup_vgg_1_xzt_stride(autoencoder_stage, modelpath_and_name)
        
    elif model_tag == "vgg_2":
        model = setup_vgg_2(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_dropout":
        model = setup_vgg_2_dropout(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_max":
        model = setup_vgg_2_max(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_stride":
        model = setup_vgg_2_stride(autoencoder_stage, modelpath_and_name)
    
    elif model_tag == "vgg_3" or model_tag=="vgg_3_eps":
        options_dict["filter_base_version"]="std"
        model = setup_vgg_3(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_3_xztc":
        options_dict["filter_base_version"]="xztc"
        model = setup_vgg_3(autoencoder_stage, options_dict, modelpath_and_name)  
    elif model_tag == "vgg_3_small":
        model = setup_vgg_3_small(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_verysmall":
        model = setup_vgg_3_verysmall(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_dropout":
        model = setup_vgg_3_dropout(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_max":
        model = setup_vgg_3_max(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_stride":
        model = setup_vgg_3_stride(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_stride_noRelu":
        model = setup_vgg_3_stride_noRelu(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_3_reg":
        model = setup_vgg_3_reg(autoencoder_stage, modelpath_and_name)
        
    elif model_tag == "vgg_4_6c":
        model = setup_vgg_4_6c(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_ConvAfterPool":
        model = setup_vgg_4_ConvAfterPool(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_6c_scale":
        model = setup_vgg_4_6c_scale(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_8c":
        model = setup_vgg_4_8c(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_10c":
        model = setup_vgg_4_10c(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_10c_smallkernel":
        model = setup_vgg_4_10c_smallkernel(autoencoder_stage, modelpath_and_name)   
    elif model_tag == "vgg_4_10c_triple":
        model = setup_vgg_4_10c_triple(autoencoder_stage, modelpath_and_name) 
    elif model_tag == "vgg_4_10c_triple_same_structure":
        model = setup_vgg_4_10c_triple_same_structure(autoencoder_stage, modelpath_and_name) 
    elif model_tag == "vgg_4_7c_less_filters":
        model = setup_vgg_4_7c_less_filters(autoencoder_stage, modelpath_and_name) 
    elif model_tag == "vgg_4_7c_same_structure":
        model = setup_vgg_4_7c_same_structure(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_15c":
        model = setup_vgg_4_15c(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_4_30c":
        model = setup_vgg_4_30c(autoencoder_stage, modelpath_and_name)

    elif model_tag == "vgg_5_2000":
        model = setup_vgg_5_2000(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_5_4000":
        model = setup_vgg_5_4000(autoencoder_stage, options_dict, modelpath_and_name)
    
    elif model_tag == "vgg_5_picture":
        model = setup_vgg_5_picture(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_5_channel":
        model = setup_vgg_5_channel(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_5_morefilter":
        model = setup_vgg_5_morefilter(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_5_morefilter_double":
        options_dict["layer_version"]="double"
        model = setup_vgg_5_morefilter(autoencoder_stage, options_dict, modelpath_and_name)
        
    elif model_tag == "vgg_5_200":
        options_dict["filter_base_version"]="standard"
        model = setup_vgg_5_200(autoencoder_stage, options_dict, modelpath_and_name)   
    elif model_tag == "vgg_5_200_large":
        options_dict["filter_base_version"]="large"
        model = setup_vgg_5_200(autoencoder_stage, options_dict, modelpath_and_name)  
    elif model_tag == "vgg_5_200_deep":
        model = setup_vgg_5_200_deep(autoencoder_stage, options_dict, modelpath_and_name)  
    elif model_tag == "vgg_5_200_dense":
        model = setup_vgg_5_200_dense(autoencoder_stage, options_dict, modelpath_and_name) 
    elif model_tag == "vgg_5_200_small":
        options_dict["filter_base_version"]="small"
        model = setup_vgg_5_200(autoencoder_stage, options_dict, modelpath_and_name) 
    elif model_tag == "vgg_5_200_shallow":
        model = setup_vgg_5_200_shallow(autoencoder_stage, options_dict, modelpath_and_name)  
    elif model_tag == "vgg_5_200_xztc":
        options_dict["filter_base_version"]="xztc"
        model = setup_vgg_5_200(autoencoder_stage, options_dict, modelpath_and_name)   
    elif model_tag == "vgg_5_xztc":
        model = setup_vgg_5_xztc(autoencoder_stage, options_dict, modelpath_and_name) 
    
    elif model_tag == "vgg_5_64":
        model = setup_vgg_5_64(autoencoder_stage, options_dict, modelpath_and_name) 
        
    elif model_tag == "vgg_5_32":
        model = setup_vgg_5_32(autoencoder_stage, options_dict, modelpath_and_name)


    elif model_tag == "channel_vgg":
        options_dict["number_of_filters_in_input"]=31
        model = setup_channel_vgg(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "channel_vgg_xyz":
        options_dict["number_of_filters_in_input"]=1
        model = setup_channel_vgg(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "channel_1n":
        options_dict["neurons_in_bottleneck"]=1
        options_dict["model_type"]=0
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)    
    elif model_tag == "channel_5n":
        options_dict["neurons_in_bottleneck"]=5
        options_dict["model_type"]=0
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name) 
    elif model_tag == "channel_5n_small":
        options_dict["neurons_in_bottleneck"]=5
        options_dict["model_type"]=1
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)  
    elif model_tag == "channel_3n":
        options_dict["neurons_in_bottleneck"]=3
        options_dict["model_type"]=2
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name) 
        
    elif model_tag == "channel_2n":
        options_dict["neurons_in_bottleneck"]=2
        options_dict["model_type"]=3
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "channel_3n_m3":
        options_dict["neurons_in_bottleneck"]=3
        options_dict["model_type"]=3
        options_dict["make_stateful"]=True
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "channel_5n_m3":
        options_dict["neurons_in_bottleneck"]=5
        options_dict["model_type"]=3
        options_dict["make_stateful"]=True
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "channel_10n_m3":
        options_dict["neurons_in_bottleneck"]=10
        options_dict["model_type"]=3
        options_dict["make_stateful"]=True
        model = setup_channel(autoencoder_stage, options_dict, modelpath_and_name)
        
    elif model_tag == "channel_tiny":
        options_dict["neurons_in_bottleneck"]=5
        model = setup_channel_tiny(autoencoder_stage, options_dict, modelpath_and_name)
    
    elif model_tag == "vgg_6_200_advers":
        model = setup_vgg_6_200_advers(autoencoder_stage, options_dict, modelpath_and_name)
    elif model_tag == "vgg_6_2000_advers":
        model = setup_vgg_6_2000_advers(autoencoder_stage, options_dict, modelpath_and_name)
    
    
    else:
        raise Exception('Model tag not available: '+ model_tag)
    return model


def print_model_blockwise(model):
    for layer in model.layers:
        if isinstance(layer, Conv3D) or isinstance(layer, Conv3DTranspose) or isinstance(layer, AveragePooling3D) or isinstance(layer, UpSampling3D):
            print(layer.name, "\t", layer.output_shape[1:])

    
            
if __name__=="__main__":
    model=setup_model(model_tag="vgg_6_2000_advers", autoencoder_stage=0, modelpath_and_name=None, 
                      additional_options="")
    #model.summary()
    conv_layer_indices = []
    for layer_index, layer in enumerate(model.layers):
        if isinstance(layer, Conv3D):
            conv_layer_indices.append(layer_index)
    print(len(conv_layer_indices))

"""
import numpy as np
import matplotlib.pyplot as plt

def print_layer_output_info(model):
    shape=0
    print("Depth\tSize\tShape")
    for depth,layer in enumerate(model.layers):
        shape=layer.output_shape[1:]
        print(depth, "\t", prod(shape),"\t", shape)

def print_layer_output_info_relev(model):
    skip_these_layers=("batch_normalization","activation","zero_padding", "dropout")#, "up_sampling", "average_pooling", "lambda", "max_pooling")
    shape=0
    size_array=[]
    print("Depth\tSize\tShape")
    for depth,layer in enumerate(model.layers):
        if not any(name in layer.name for name in skip_these_layers):
            shape=layer.output_shape[1:]
            print(depth, "\t", np.prod(shape),"\t", shape,"\t", layer.name)
            size_array.append(np.prod(shape))
    return size_array

def plot_model_output_size(model_array):
    for i,mo in enumerate(model_array):
        arr=print_layer_output_info_relev(mo)
        x_data=np.linspace(0.5,len(arr)-0.5,len(arr))
        plt.step(x_data, arr, where='mid', label=str(i+1))
    plt.legend()
    plt.show()

#For testing purposes
model = setup_model("vgg_3", 0)
#model3 = setup_model("vgg_1_xzt_stride", 0)

plot_model_output_size([model,])
tot_neu=0
for layer in model.layers:
    if "conv3d" in layer.name:
        size = np.prod(layer.output_shape[1:])
        tot_neu+=size
        print(layer.name, tot_neu)
print(tot_neu)
"""

"""
#from numpy import prod
def summ_model(model):
    shape=0
    deep=0
    print("Depth\tSize\tShape")
    for depth,layer in enumerate(model.layers):
        deep=depth
        #shape=layer.output_shape[1:]
        #print(depth, "\t", prod(shape),"\t", shape)
    print(deep)
"""

#Outdated:
"""
def setup_vgg_like(autoencoder_stage, modelpath_and_name=None):
    #a vgg-like autoencoder, witht lots of convolutional layers
    #tag: vgg_0
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
    
    inputs = Input(shape=(11,13,18,1))
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train)(inputs)
    x = Conv3D(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=train)(x)
    #10x12x16 x 32
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 32
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    #4x4x6 x 64
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x2x3 x 64

    if autoencoder_stage == 0:
        #The Decoder part:
        
        #2x2x3 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x4x6 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
        #5x6x8 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = UpSampling3D((2, 2, 2))(x)
        #10x12x16 x 64
        x = Conv3DTranspose(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
        #11x13x18 x 32
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        decoded = Conv3DTranspose(filters=1, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else:
        #Replacement for the decoder part for supervised training:
        
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_conv_model_API():
    #Wie der autoencoder im sequential style, nur mit API
    
    inputs = Input(shape=(11,13,18,1))
    x = Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu')(inputs)
    #10x12x16 x 16
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 16
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #3x4x6 x 8
    encoded = Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
    #2x2x4 x 4
    
    
    #2x2x4 x 4
    x = Conv3DTranspose(filters=8, kernel_size=(2,3,3), padding='valid', activation='relu' )(encoded)
    #3x4x6 x 8
    x = Conv3DTranspose(filters=16, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #5x6x8 x 16
    x = UpSampling3D((2, 2, 2))(x)
    #10x12x16 x 16
    decoded = Conv3DTranspose(filters=1, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
    #Output 11x13x18 x 1
    
    autoencoder = Model(inputs, decoded)
    return autoencoder
"""
