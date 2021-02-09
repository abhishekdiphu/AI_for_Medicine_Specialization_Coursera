
import argparse
import importlib
## importing all the packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy

from integrated_grad import *
import util
from util import *
from  helper import *

import scikitplot
import sklearn
from sklearn.metrics import confusion_matrix


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




def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=32, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    print("done")
    
    return generator






def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=32, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="images", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator




train_df = pd.read_csv("nih/train.csv")
valid_df = pd.read_csv("nih/test.csv")
test_df = pd.read_csv("nih/train.csv")
train_df.head(5)
labels = ['tuberculosis']







IMAGE_DIR = "nih/ChinaSet_AllFiles/CXR_png/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "images", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "images", labels)


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
print("freq_pos", freq_pos)
print("freq_pos", freq_neg)
pos_weights = freq_neg
neg_weights = freq_pos
print("length of pos_weights :",len(pos_weights))
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights
print("pos_contribution", pos_contribution)
print("neg_contribution", neg_contribution)

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)



# create the base pre-trained model
base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(lr =0.0002), loss=BinaryCrossentropy(),metrics =['accuracy'])

#model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=5,
    mode = max,
    verbose=1)

training = False
if training :
        history = model.fit_generator(train_generator, 
                                      validation_data=valid_generator,
                                      steps_per_epoch=None, 
                                      validation_steps=6, 
                                      epochs = 20,
                                      callbacks = [early_stopping])




        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training Loss Curve")
        plt.savefig("training_loss", dpi =100)

        model.save_weights("./nih/den_model.h5")

model.load_weights("./nih/den_model01.h5")
print("length of the generator:" ,len(train_generator) )
predicted_vals = model.predict(train_generator, steps = len(train_generator))

print(test_generator.labels.shape)

#true =  np.argmax(test_generator.labels,axis=1)
true =   train_generator.labels
#print(true)
#preds = np.argmax(predicted_vals, axis =1)

preds = predicted_vals#.reshape(-1)

preds[preds <= 0.5] = 0.
preds[preds > 0.5] = 1.

print(confusion_matrix(true, preds , normalize= None))
scikitplot.metrics.plot_confusion_matrix(true, preds , normalize= True, figsize=(10,10), cmap='inferno_r')
plt.savefig("confusion_matrix.png")
plt.close()


cl_report = sklearn.metrics.classification_report(true, preds, target_names  =['tuberculosis' , 'normal'])
print("the classification report : \n" , cl_report)



df = pd.read_csv("nih/dataset_06.csv")
IMAGE_DIR = "nih/ChinaSet_AllFiles/CXR_png/"

# only show the lables with top 4 AUC
auc_rocs = util.get_roc_curve(['tuberculosis'], predicted_vals, test_generator)
plt.savefig("auc.png")
plt.close()




image_name_1 = 'CHNCXR_0602_1.png'
image_name_2 = 'CHNCXR_0600_1.png'
labeling = ['tuberculosis' ,'normal']
labels_to_show = np.take(labeling, np.argsort(auc_rocs)[::-1])[:4]

util.compute_gradcam(model, image_name_1, IMAGE_DIR, df, labels, labels_to_show)
plt.savefig(image_name_1)
plt.close()
util.compute_gradcam(model,image_name_2 , IMAGE_DIR, df, labels, labels_to_show)
plt.savefig(image_name_2)
plt.close()







integrated_g = False

if integrated_g :
        # 1. Convert the image to numpy array
        img = get_img_array('/content/nih/ChinaSet_AllFiles/CXR_png/CHNCXR_0001_0.png')

        # 2. Keep a copy of the original image
        orig_img = np.copy(img[0]).astype(np.uint8)

        # 3. Preprocess the image
        img_processed = tf.cast(preprocess_input(img), dtype=tf.float32)

        print(img_processed.shape)
        # 4. Get model predictions
        preds = model.predict(img_processed, steps =1)
        top_pred_idx = tf.argmax(preds[0])
        #print("Predicted:", top_pred_idx, decode_predictions(preds, top=1)[0])

        # 5. Get the gradients of the last layer for the predicted label
        grads = get_gradients(img_processed, top_pred_idx=top_pred_idx, model = model)

        # 6. Get the integrated gradients
        igrads = random_baseline_integrated_gradients(
            np.copy(orig_img), top_pred_idx=top_pred_idx, num_steps=2, num_runs=2, model =model
        )


        # 7. Process the gradients and plot
        vis = GradVisualizer()
        vis.visualize(
            image=orig_img,
            gradients=grads[0].numpy(),
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=99,
            clip_below_percentile=0,
        )

        vis.visualize(
            image=orig_img,
            gradients=grads[0].numpy(),
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=95,
            clip_below_percentile=28,
            morphological_cleanup=True,
            outlines=True,
        )
