"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape

import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import keras
from mpl_toolkits.mplot3d import Axes3D 


def get_sub_volume(image, label, 
                   orig_x = 512, orig_y = 512, orig_z = 90, 
                   output_x = 64, output_y = 64, output_z = 64,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    """
    Extract random sub-volume from original images.
    Args:
        image (np.array): original image, 
            of shape (orig_x, orig_y, orig_z, num_channels)
        label (np.array): original label. 
            labels coded using discrete values rather than
            a separate dimension, 
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        num_classes (int): number of class labels
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction 
            of the sample which can be the background
    returns:
        X (np.array): sample of original image of dimension 
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension 
            (num_classes, output_x, output_y, output_z)
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    tries = 0
    
    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x + 1) 
        start_y = np.random.randint(0 ,orig_x - output_x + 1) 
        start_z = np.random.randint(0 ,orig_z - output_z + 1) 

        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
        # One-hot encode the categories.
        # This adds a 4th dimension, 'num_classes'
        # (output_x, output_y, output_z, num_classes)
        y = keras.utils.to_categorical(y, num_classes= num_classes)

        # compute the background ratio
        bgrd_ratio = np.sum(y[: , : , : , 0]) / (output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:

            # make copy of the sub-volume
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z])
            
            # change dimension of X
            # from (x_dim, y_dim, z_dim, num_channels)
            # to (num_channels, x_dim, y_dim, z_dim)
            #X = np.moveaxis(X , 3 , 0)

            # change dimension of y
            # from (x_dim, y_dim, z_dim, num_classes)
            # to (num_classes, x_dim, y_dim, z_dim)
            y = np.moveaxis(y , 3 , 0)

            ### END CODE HERE ###
            
            # take a subset of y that excludes the background class
            # in the 'num_classes' dimension
            y = y[1:, :, :, :]
    
            return X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    count = 1
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))
        #print(label.shape)
        #print(image.shape)
        #print(f'With the unique values: {np.unique(label)}')

        image, label = get_sub_volume(image, label, 
                   orig_x = image.shape[0], orig_y =image.shape[1], orig_z = image.shape[2], 
                   output_x = 256, output_y = 256, output_z = 40,
                   num_classes = 2, max_tries = 1000, 
                   background_threshold=0.99)

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        # <YOUR CODE GOES HERE>
        image = image / image.max()
        label =np.reshape(label, (256,256,40))
        #print("label shape " , label.shape)
        #print(image.shape)
        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        
        label = med_reshape(label, new_shape=(label.shape[1], y_shape, z_shape)).astype(int)
        #print("label shape " , label.shape)
        print("image shape", image.shape)
        # TASK: Why do we need to cast label to int?
        # ANSWER: Because the loss function needs to do math operations on the label.
        idx = 12
        #print("image no :" , count)
        count = count +1
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image[:,:,idx], cmap='gray')

        plt.subplot(1,2,2)
        plt.imshow(label[:,:,idx], cmap='gray')
        plt.title(f"axial plane idx={f}")
        plt.savefig('image')
        plt.close()
        image = np.moveaxis(image , 2 , 0)
        label = np.moveaxis(label , 2 , 0)
        #print("label shape " , label.shape)
        #print("image shape", image.shape)
        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
