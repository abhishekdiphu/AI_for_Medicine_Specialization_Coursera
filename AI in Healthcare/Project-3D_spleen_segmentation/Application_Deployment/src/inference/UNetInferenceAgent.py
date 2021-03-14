"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape
from PIL import Image

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=128):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=2)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        print("volume shape :" , volume.shape)
        volume = self.get_sub_volume(volume ,orig_x = 512, orig_y = 512, orig_z = volume.shape[2], 
                       output_x = 128, output_y = 128, output_z = volume.shape[2],
                       num_classes = 2, max_tries = 1000, 
                       background_threshold=0.99)
        

        volume = med_reshape(volume, (volume.shape[0], self.patch_size, volume.shape[2]))
        print("volume shape :" , volume.shape)
        
        return self.single_volume_inference(volume)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []
        print("volume shape :" , volume.shape)
        volume = np.swapaxes(volume, 0, -1)
        print("volume shape :" , volume.shape)
        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        outputs = self.model(torch.tensor(volume).type(torch.FloatTensor).unsqueeze(1).to(self.device)).cpu().detach()
        _, slices = torch.max(outputs, 1)
        
        return slices.numpy()
    
    
    def get_sub_volume(self, image , orig_x = 512, orig_y = 512, orig_z = 90, 
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
            #y = label[start_x: start_x + output_x,
            #          start_y: start_y + output_y,
            #          start_z: start_z + output_z]

            # One-hot encode the categories.
            # This adds a 4th dimension, 'num_classes'
            # (output_x, output_y, output_z, num_classes)
            #y = keras.utils.to_categorical(y, num_classes= num_classes)

            # compute the background ratio
            #bgrd_ratio = np.sum(y[: , : , : , 0]) / (output_x * output_y * output_z)

            # increment tries counter
            #tries += 1

            # if background ratio is below the desired threshold,
            # use that sub-volume.
            # otherwise continue the loop and try another random sub-volume
            #if bgrd_ratio < background_threshold:

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
                #y = np.moveaxis(y , 3 , 0)

                ### END CODE HERE ###

                # take a subset of y that excludes the background class
                # in the 'num_classes' dimension
                #y = y[1:, :, :, :]

            return X#, y

        # if we've tried max_tries number of samples
        # Give up in order to avoid looping forever.
        print(f"Tried {tries} times to find a sub-volume. Giving up...")
