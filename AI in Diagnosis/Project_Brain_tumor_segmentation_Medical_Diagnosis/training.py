import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K 

import utils.util as util
from utils.helper_functions import *
import  models.model  as model

import argparse
import importlib
## importing all the packages


parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=False, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=350, help='frequency of saving the model')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam or RMS-Prop)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')
parser.add_argument('--gpu_device', type=int, default=0, help='GPU device which needs to be used for computation')
parser.add_argument('--validation_sample_size', type=int, default=1, help='size of validation sample')
parser.add_argument('--validate_every', type=int, default=5, help='frequency of evaluating on validation set')

parser.add_argument('--path', default='/datasets/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.804)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=2)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--dataset', type=str, required=False, choices=['mpii', 'lsp', 'medical'])

args = parser.parse_args()


# set home directory and data directory
HOME_DIR = "/content/BraTs-Data/BraTs-Data/"
DATA_DIR = HOME_DIR

def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    
    return image, label


image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")
image = util.get_labeled_image(image, label)

util.plot_image_grid(image)


image, label = load_case(DATA_DIR + "imagesTr/BRATS_001.nii.gz", DATA_DIR + "labelsTr/BRATS_001.nii.gz")
X, y = get_sub_volume(image, label)
# enhancing tumor is channel 2 in the class label
# you can change indexer for y to look at different classes
util.visualize_patch(X[0, :, :, :], y[1])




X_norm = standardize(X)
print("standard deviation for a slice should be 1.0")
print(f"stddv for X_norm[0, :, :, 0]: {X_norm[0,:,:,0].std():.2f}")
util.visualize_patch(X_norm[0, :, :, :], y[0])




model = model.unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])


base_dir = HOME_DIR + "processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)

# Get generators for training and validation sets
train_generator = util.VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=0)
valid_generator = util.VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=0)




training = True
if training = True:
        steps_per_epoch = 20
        n_epochs=50
        validation_steps = 20
        history = model.fit_generator(generator=train_generator,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=n_epochs,
                                        use_multiprocessing=True,
                                        validation_data=valid_generator,
                                        validation_steps=validation_steps)




        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training Loss Curve")
        plt.savefig("training_dice_loss", dpi =100)
        model.save_weights(base_dir + 'my_model_pretrained.hdf5')
