# -*- coding: utf-8 -*-
"""
Contains Definitions of setup_vgg_2, setup_vgg_2_max, setup_vgg_2_stride, setup_vgg_2_dropout

Vgg-like autoencoder-networks with 7+7 convolutional layers w/ batch norm; ca 720k free params
Input Format: 11x18x50 (XZT DATA)

autoencoder_stage: Type of training/network
    0: autoencoder
    1: encoder+ from autoencoder w/ frozen layers
    2: encoder+ from scratch, completely unfrozen
    
If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
These layers are frozen then.
The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)

modelpath_and_name is used to load the encoder part for supervised training, 
and only needed if make_autoencoder==False
    
"""
from keras.models import Model
from keras.layers import Activation, Input, Dropout, Dense, Flatten, Conv3D, MaxPooling3D, UpSampling3D,BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
from keras import backend as K

from util.custom_layers import MaxUnpooling3D

#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0):
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    x = Activation('relu', trainable=trainable)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    return x

def convT_block(inp, filters, kernel_size, padding, channel_axis, strides=(1,1,1), dropout=0.0):
    x = Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False)(inp)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    return x

def setup_vgg_2(autoencoder_stage, modelpath_and_name=None):
    #713k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x)
    #11x18x25
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(2,3,2), padding="valid", trainable=train, channel_axis=channel_axis)  
    #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(2,3,3), padding="valid", trainable=train, channel_axis=channel_axis)
    #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x3x5
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x6x10
        x=convT_block(x, filters=64, kernel_size=(2,3,3), padding="valid", channel_axis=channel_axis)
        #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x = UpSampling3D((2, 2, 2))(x)
        #10x16x24
        x=convT_block(x, filters=32, kernel_size=(2,3,2), padding="valid", channel_axis=channel_axis)
        #11x18x25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x = UpSampling3D((1, 1, 2))(x)
        #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_2_dropout(autoencoder_stage, modelpath_and_name=None):
    #713k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    dropout_rate=0.1
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x)
    #11x18x25
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(2,3,2), padding="valid",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)  
    #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(2,3,3), padding="valid",dropout=dropout_rate, trainable=train, channel_axis=channel_axis)
    #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x3x5
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x6x10
        x=convT_block(x, filters=64, kernel_size=(2,3,3), padding="valid",dropout=dropout_rate, channel_axis=channel_axis)
        #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, channel_axis=channel_axis)
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, channel_axis=channel_axis)
        x = UpSampling3D((2, 2, 2))(x)
        #10x16x24
        x=convT_block(x, filters=32, kernel_size=(2,3,2), padding="valid",dropout=dropout_rate, channel_axis=channel_axis)
        #11x18x25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, channel_axis=channel_axis)
        x = UpSampling3D((1, 1, 2))(x)
        #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same",dropout=dropout_rate, channel_axis=channel_axis)
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_2_max(autoencoder_stage, modelpath_and_name=None):
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    #11x18x50
    x = MaxPooling3D((1, 1, 2), padding='valid')(x)
    #11x18x25
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(2,3,2), padding="valid", trainable=train, channel_axis=channel_axis)    
    #10x16x24
    x = MaxPooling3D((2, 2, 2), padding='valid')(x)
    #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=64, kernel_size=(2,3,3), padding="valid", trainable=train, channel_axis=channel_axis)
    #4x6x10
    encoded = MaxPooling3D((2, 2, 2), padding='valid')(x)
    #2x3x5 x 64    (=1920 = 19.4 % org size)

    if autoencoder_stage == 0:
        #The Decoder part:
        print("Loading Decoder")
        #2x3x5 x 64
        x = MaxUnpooling3D(encoded)
        #4x6x10
        x=convT_block(x, filters=64, kernel_size=(2,3,3), padding="valid", channel_axis=channel_axis)
        #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x = MaxUnpooling3D(x)
        #10x16x24
        x=convT_block(x, filters=32, kernel_size=(2,3,2), padding="valid", channel_axis=channel_axis)
        #11x18x25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)      
        x = MaxUnpooling3D(x,(1,1,2))
        #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_2_stride(autoencoder_stage, modelpath_and_name=None):
    #like vgg2xzt, but with stride>1 instead of pooling
    #750k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis)
    #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), trainable=train, channel_axis=channel_axis)
    #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x)
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis)
    #10x16x24
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", strides=(2,2,2), trainable=train, channel_axis=channel_axis)
    #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x)
    #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis)
    #4x6x10
    encoded=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", strides=(2,2,2), trainable=train, channel_axis=channel_axis)
    #2x3x5

    if autoencoder_stage == 0:
        #The Decoder part:
        #2x3x5 x 64
        x=convT_block(encoded, filters=64, kernel_size=(3,3,3), padding="same", strides=(2,2,2), channel_axis=channel_axis)
        #4x6x10
        x = ZeroPadding3D(((1,2),(2,2),(2,2)))(x)
        #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis)
        #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(2,2,2), channel_axis=channel_axis)
        #10x16x24
        x = ZeroPadding3D(((1,2),(2,2),(1,2)))(x)
        #13x20x27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis)
        #11x18x25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), channel_axis=channel_axis)
        #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
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
    