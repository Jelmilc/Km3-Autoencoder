#!/bin/bash -l
#
# job name for PBS, out and error files will also have this name + an id
#PBS -N vgg3_test
#
# first non-empty non-comment line ends PBS options


# Demo input file for autoencoder training

# Tag for the model used; Identifies both autoencoder and encoder
# This also defines the name of the folder where everything is saved
# can append '-XXX' for model version, which is saved in its own folder
modeltag="vgg_5_200"
#How many additinal epochs the network will be trained for by executing this script:
runs=100
#Type of training/network
# 0: autoencoder
# 1: encoder+dense from autoencoder w/ frozen layers
# 2: encoder+dense from scratch, completely unfrozen
# 3: Parallel supervised training (load in new frozen encoder epoch according to schedule)
# 4: Unfreeze training: Like stage 1, but conv blocks are unfrozen every few epochs.
autoencoder_stage=2
#Define starting epoch of autoencoder model
#(stage 1: the frozen encoder part of which autoencoder epoch to use)
# if -1, the latest epoch is automatically looked for
# not used for stage 2 and 3
autoencoder_epoch=-1
#If in encoder stage (1 or 2), encoder_epoch is used to identify a possibly
#existing supervised-trained encoder network; not used for stage 0
# -1: Latest epoch
encoder_epoch=-1
#Define what the supervised encoder network is trained for, and how many neurons are in the output
#This also defines the name of the saved model
# class type names are currently "up_down", "muon-CC_to_elec-CC", "energy"
class_type_bins=2
class_type_name="muon-CC_to_elec-CC"
#Whether to use a zero-center image or not (1 or 0)
#if none exists, a new one will be generated and saved
zero_center=1
#Verbose bar during training?
#0: silent, 1:verbose, 2: one log line per epoch
verbose=2
# which dataset to use; see get_dataset_info.py
# x  y  z  t  c
# 11 13 18 50 31
#additional options can be appended via -XXX-YYY, e.g. "xzt-filesize=0.3" (see get_dataset_info.py)
dataset="xzt"
#Initial Learning rate, usually 0.001
# negative lr = take the absolute as the lr at epoch 1, and apply the decay epoch times
learning_rate=-0.001
#lr_decay can either be a float, e.g. 0.05 for 5% decay of lr per epoch,
#or it can be a string like s1 for lr schedule 1.
#available lr schedules are listed in run_autoencoder.py
learning_rate_decay="s1"

# exponent of epsilon for the adam optimizer (actual epsilon is 10^this)
# Autoencoder: should be -1
epsilon=-1
# lambda compatibility mode (0 or 1): Load the model manually from model definition,
# and insert the weigths; can be used to overwrite parameters of the optimizer
lambda_comp=0
#optimizer to use, either "adam" or "sgd"
optimizer="adam"
# Additional options for the model, see model_definitions.py; "None" for no options
# e.g. "dropout=0.3", "unlock_BN", ...
options="dropout=0.1"

# allows to create multiple supervised trainings from the same AE model
# the version is added to the filename as a string, so empty string is versionless
encoder_version=""
# Defines the loss that is used for training during autoencoder stage
# usually mse, but other options are available (see run_autoencoder.py)
ae_loss_name="mse"
# Defines the loss that is used for training during supervised stages
# can be categorical_crossentropy, mse or mae, or auto
# auto will choose the loss based on the number of output neurons:
#    1 neuron : mae
#    2 neurons: categorical_crossentropy
supervised_loss="auto"
# Define a modelpath to initialize the weights from the current model from.
# Can not be used for parallel training (stage 3)
#init_model=None



cd $WOODYHOME/code
#Setup environment
source env_cnn.sh
#Execute training
python scripts/run_autoencoder.py $modeltag $runs $autoencoder_stage $autoencoder_epoch $encoder_epoch $class_type_bins $class_type_name $zero_center $verbose $dataset $learning_rate $learning_rate_decay $epsilon $lambda_comp $optimizer $options $encoder_version --ae_loss_name=$ae_loss_name --supervised_loss=$supervised_loss

