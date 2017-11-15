# -*- coding: utf-8 -*-
import h5py
import numpy as np
from keras import backend as K
import warnings

from Loggers import *

"""
train_and_test_model(model, modelname, train_files, test_files, batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=0,
                         shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None):
"""

def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, lr_decay, tb_logger, swap_4d_channels, save_path, is_autoencoder, verbose):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    epoch += 1
    if epoch > 1 and lr_decay > 0:
        lr *= 1 - float(lr_decay)
        K.set_value(model.optimizer.lr, lr)
        print ('Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch) + ' (minus ' + str(lr_decay) + ')')

    fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch, shuffle, swap_4d_channels, is_autoencoder=is_autoencoder, n_events=None, tb_logger=tb_logger, save_path=save_path, verbose=verbose)
    #fit_model speichert model ab unter ("models/tag/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
    #evaluate model evaluated und printet es in der konsole und in file
    evaluation = evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=None, is_autoencoder=is_autoencoder)

    with open(save_path+"trained_" + modelname + '_test.txt', 'a') as test_file:
        if is_autoencoder==False:
            #loss and accuracy
            test_file.write('\n{0}\t{1}\t{2}\t{3}'.format(epoch, lr, evaluation[0], evaluation[1]))
        else:
            #For autoencoders: only loss
            test_file.write('\n{0}\t{1}\t{2}'.format(epoch, lr, evaluation))
    return lr
            



def fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, save_path, is_autoencoder, verbose, n_events=None, tb_logger=False):
    """
    Trains a model based on the Keras fit_generator method.
    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param str modelname: Name of the model.
    :param list train_files: list of tuples that contains the testfiles and their number of rows (filepath, f_size).
    :param list test_files: list of tuples that contains the testfiles and their number of rows for the tb_callback.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in both the train- and test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Epoch of the model if it has been trained before.
    :param bool shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
    """

    validation_data, validation_steps, callbacks = None, None, None

    for i, (f, f_size) in enumerate(train_files):  # process all h5 files, full epoch
        if epoch > 1 and shuffle is True: # just for convenience, we don't want to wait before the first epoch each time
            print ('Shuffling file ', f, ' before training in epoch ', epoch)
            shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
        print ('Training in epoch', epoch, 'on file ', i, ',', f)

        if n_events is not None: f_size = n_events  # for testing
        
        
        with open(save_path+"trained_" + modelname + '_epoch' + str(epoch) + '_log.txt', 'w') as log_file:
            
            if is_autoencoder == True:
                BatchLogger = NBatchLogger_Recent(display=500, logfile=log_file)
            else:
                BatchLogger = NBatchLogger_Recent_Acc(display=500, logfile=log_file)
                
            model.fit_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, is_autoencoder=is_autoencoder, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels),
                steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=verbose, max_queue_size=10,
                validation_data=validation_data, validation_steps=validation_steps, callbacks=[BatchLogger])
            model.save(save_path+"trained_" + modelname + '_epoch' + str(epoch) + '.h5') #TODO
        
        
        
def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, is_autoencoder, n_events=None,):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.
    :param ks.model.Model/Sequential model: Keras model (trained) of a neural network.
    :param list test_files: list of tuples that contains the testfiles and their number of rows.
    :param int batchsize: Batchsize that is used in the evaluate_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for evaluating.
    """
    for i, (f, f_size) in enumerate(test_files):
        print ('Testing on file ', i, ',', f)

        if n_events is not None: f_size = n_events  # for testing

        evaluation = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, is_autoencoder=is_autoencoder, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / batchsize), max_queue_size=10)
    return_message = 'Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')'
    print (return_message)
    return evaluation

        
#Copied from cnn_utilities and modified:
def generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, f_size=None, zero_center_image=None, yield_mc_info=False, swap_col=None):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated. Ideally same as the chunksize in the h5 file.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the h5 file.
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param ndarray zero_center_image: mean_image of the x dataset used for zero-centering.
    :param bool yield_mc_info: Specifies if mc-infos (y_values) should be yielded as well.
                               The mc-infos are used for evaluation after training and testing is finished.
    :param bool swap_col: Specifies, if the index of the columns for xs should be swapped. Necessary for 3.5D nets.
                          Currently available: 'yzt-x' -> [3,1,2,0] from [0,1,2,3]
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info if yield_mc_info=True).
    """
    dimensions = get_dimensions_encoding(n_bins, batchsize)

    while 1:
        f = h5py.File(filepath, "r")
        if f_size is None:
            f_size = len(f['y'])
            warnings.warn('f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                          'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

        n_entries = 0
        while n_entries <= (f_size - batchsize):
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize]
            xs = np.reshape(xs, dimensions).astype(np.float32)

            if swap_col is not None:
                swap_4d_channels_dict = {'yzt-x': [3,1,2,0]}
                xs[:, swap_4d_channels_dict[swap_col]] = xs[:, [0,1,2,3]]

            if zero_center_image is not None: xs = np.subtract(xs, zero_center_image) # if swap_col is not None, zero_center_image is already swapped
            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1])) #TODO simplify with (y_values, y_values.shape) ?
            
            # we have read one more batch from this file
            n_entries += batchsize
            
            #Modified for autoencoder:
            if is_autoencoder == True:
                output = (xs, xs)
                yield output
                
            else:
                ys = np.zeros((batchsize, class_type[0]), dtype=np.float32)
                # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
                for c, y_val in enumerate(y_values): # Could be vectorized with numba, or use dataflow from tensorpack
                    ys[c] = encode_targets(y_val, class_type)
                output = (xs, ys) if yield_mc_info is False else (xs, ys) + (y_values,)
                yield output
            
        f.close() # this line of code is actually not reached if steps=f_size/batchsize
        
#Copied, removed print
def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    if n_bins_x == 1:
        if n_bins_y == 1:
            #print 'Using 2D projected data without dimensions x and y'
            dimensions = (batchsize, n_bins_z, n_bins_t, 1)
        elif n_bins_z == 1:
            #print 'Using 2D projected data without dimensions x and z'
            dimensions = (batchsize, n_bins_y, n_bins_t, 1)
        elif n_bins_t == 1:
            #print 'Using 2D projected data without dimensions x and t'
            dimensions = (batchsize, n_bins_y, n_bins_z, 1)
        else:
            #print 'Using 3D projected data without dimension x'
            dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)

    elif n_bins_y == 1:
        if n_bins_z == 1:
            #print 'Using 2D projected data without dimensions y and z'
            dimensions = (batchsize, n_bins_x, n_bins_t, 1)
        elif n_bins_t == 1:
            #print 'Using 2D projected data without dimensions y and t'
            dimensions = (batchsize, n_bins_x, n_bins_z, 1)
        else:
            #print 'Using 3D projected data without dimension y'
            dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)

    elif n_bins_z == 1:
        if n_bins_t == 1:
            #print 'Using 2D projected data without dimensions z and t'
            dimensions = (batchsize, n_bins_x, n_bins_y, 1)
        else:
            #print 'Using 3D projected data without dimension z'
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)

    elif n_bins_t == 1:
        #print 'Using 3D projected data without dimension t'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    else:
        #print 'Using full 4D data'
        dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions

        
#Copied unchanged from cnn_utilities
def encode_targets(y_val, class_type):
    """
    Encodes the labels (classes) of the images.
    :param ndarray(ndim=1) y_val: Array that contains ALL event class information for one event.
           ---------------------------------------------------------------------------------------------------------------------------
           Current content: [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
           ---------------------------------------------------------------------------------------------------------------------------
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :return: ndarray(ndim=1) train_y: Array that contains the encoded class label information of the input event.
    """
    def get_class_up_down_categorical(dir_z, n_neurons):
        """
        Converts the zenith information (dir_z) to a binary up/down value.
        :param float32 dir_z: z-direction of the event_track (which contains dir_z).
        :param int n_neurons: defines the number of neurons in the last cnn layer that should be used with the categorical array.
        :return ndarray(ndim=1) y_cat_up_down: categorical y ('label') array which can be fed to a NN.
                                               E.g. [0],[1] for n=1 or [0,1], [1,0] for n=2
        """
        # analyze the track info to determine the class number
        up_down_class_value = int(np.sign(dir_z)) # returns -1 if dir_z < 0, 0 if dir_z==0, 1 if dir_z > 0

        if up_down_class_value == 0:
            print ('Warning: Found an event with dir_z==0. Setting the up-down class randomly.')
            #TODO maybe [0.5, 0.5], but does it make sense with cat_crossentropy?
            up_down_class_value = np.random.randint(2)

        if up_down_class_value == -1: up_down_class_value = 0 # Bring -1,1 values to 0,1

        y_cat_up_down = np.zeros(n_neurons, dtype='float32')

        if n_neurons == 1:
            y_cat_up_down[0] = up_down_class_value # 1 or 0 for up/down
        else:
            y_cat_up_down[up_down_class_value] = 1 # [0,1] or [1,0] for up/down

        return y_cat_up_down


    def convert_particle_class_to_categorical(particle_type, is_cc, num_classes=4):
        """
        Converts the possible particle types (elec/muon/tau , NC/CC) to a categorical type that can be used as tensorflow input y
        :param int particle_type: Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
        :param int is_cc: Specifies the interaction channel. 0 = NC, 1 = CC.
        :param int num_classes: Specifies the total number of classes that will be discriminated later on by the CNN. I.e. 2 = elec_NC, muon_CC.
        :return: ndarray(ndim=1) categorical: returns the categorical event type. I.e. (particle_type=14, is_cc=1) -> [0,0,1,0] for num_classes=4.
        """
        if num_classes == 4:
            particle_type_dict = {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
        else:
            raise ValueError('A number of classes !=4 is currently not supported!')

        category = int(particle_type_dict[(abs(particle_type), is_cc)])
        categorical = np.zeros(num_classes, dtype='int8') # TODO try bool
        categorical[category] = 1

        return categorical


    if class_type[1] == 'muon-CC_to_elec-NC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32') # 1 ([0], [1]) or 2 ([0,1], [1,0]) neurons

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-NC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[0]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'muon-CC_to_elec-CC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32')

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-CC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[1]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'up_down':
        #supports both 1 or 2 neurons at the cnn softmax end
        train_y = get_class_up_down_categorical(y_val[7], class_type[0])

    else:
        print ("Class type " + str(class_type) + " not supported!")
        return y_val

    return train_y
        
        
#Kopiert von utilities/input utilities:
def h5_get_number_of_rows(h5_filepath):
    """
    Gets the total number of rows of the first dataset of a .h5 file. Hence, all datasets should have the same number of rows!
    :param string h5_filepath: filepath of the .h5 file.
    :return: int number_of_rows: number of rows of the .h5 file in the first dataset.
    """
    f = h5py.File(h5_filepath, 'r')
    #Bug?
    #number_of_rows = f[f.keys()[0]].shape[0]
    number_of_rows = f["x"].shape[0]
    f.close()

    return number_of_rows
        
        
#Kopiert von cnn_utilities
#Not used currently, only np.load(...)
def load_zero_center_data(train_files, batchsize, n_bins, n_gpu, swap_4d_channels=None):
    """
    Gets the xs_mean array that can be used for zero-centering.
    The array is either loaded from a previously saved file or it is calculated on the fly.
    Currently only works for a single input training file!
    :param list((train_filepath, train_filesize)) train_files: list of tuples that contains the trainfiles and their number of rows.
    :param int batchsize: Batchsize that is being used in the data.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the tran_file.
    :param int n_gpu: Number of gpu's, used for calculating the available RAM space in get_mean_image().
    :param None/str swap_4d_channels: For 4D data. Specifies if the columns in the xs_mean array should be swapped.
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    if len(train_files) > 1:
        warnings.warn('More than 1 train file for zero-centering is currently not supported! '
                      'Only the first file is used for calculating the xs_mean_array.')

    filepath = train_files[0][0]

    if os.path.isfile(filepath + '_zero_center_mean.npy') is True:
        print ('Loading an existing xs_mean_array in order to zero_center the data!')
        xs_mean = np.load(filepath + '_zero_center_mean.npy')
    else:
        print ('Calculating the xs_mean_array in order to zero_center the data!')
        dimensions = get_dimensions_encoding(n_bins, batchsize)
        xs_mean = get_mean_image(filepath, dimensions, n_gpu)

    if swap_4d_channels is not None:
        swap_4d_channels_dict = {'yzt-x': [3,1,2,0]}
        xs_mean[:, swap_4d_channels_dict['yzt-x']] = xs_mean[:, [0,1,2,3]]

    return xs_mean
        
        
        