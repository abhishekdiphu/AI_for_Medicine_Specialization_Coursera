import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from collections import OrderedDict
import torch.optim as optim
from model import *

training_volume = nib.load("data/spleen1_img.nii.gz").get_fdata()
training_label = nib.load("data/spleen1_label.nii.gz").get_fdata()



plt.imshow(training_volume[:,:,5] + training_label[:,:,5]*500, cmap="gray")
# We assume our label has one-hot encoding. Let's confirm how many distinct classes do we have in our label volume

np.unique(training_label)

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
device

# Here we will use one input channel (one image at a time) and two output channels (background and label)
unet = UNet(1, 2) 

# Move all trainable parameters to the device
unet.to(device)

# We will use Cross Entropy loss function for this one - we are performing per-voxel classification task, 
# so it should do ok.
# Later in the lesson we will discuss what are some of the other options for measuring medical image 
# segmentation performance.

loss = torch.nn.CrossEntropyLoss()

# You can play with learning rate later to see what yields best results
optimizer = optim.Adam(unet.parameters(), lr=0.001)
optimizer.zero_grad()


# By the way, how many trainable parameters does our model have? If you will be playing 
# with 3D convolutions - compare the difference between 2D and 3D versions.

total_parameters = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print("total_parameters :",total_parameters)


# This is a basic training loop. Complete the code to run the model on first 15 slices 
# of the volume (that is where the spleen segmentation is - if you include more, you run the chances of background class
# overwhelming your tiny network with cross-entropy loss that we are using)

# Set up the model for training
unet.train()

for epoch in range(0,30):
    for slice_ix in range(0,15):
        # Let's extract the slice from the volume and convert it to tensor that the model will understand. 
        # Note that we normalize the volume to 0..1 range
        slc = training_volume[:,:,slice_ix].astype(np.single)/np.max(training_volume[:,:,slice_ix])
        
        # Our model accepts a tensor of size (batch_size, channels, w, h). We have batch of 1 and one channel, 
        # So create the missing dimensions. Also move data to our device
        slc_tensor = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).to(device)
        
        # TASK: Now extract the slice from label volume into tensor that the network will accept.
        # Keep in mind, our cross entropy loss expects integers
        
        # <YOUR CODE HERE>
        # ___SOLUTION
    
        lbl = training_label[:,:,slice_ix]
        lbl_tensor = torch.from_numpy(lbl).unsqueeze(0).long().to(device)
        # ___SOLUTION

        # Zero-out gradients from the previous pass so that we can start computation from scratch for this backprop run
        optimizer.zero_grad()
        
        # Do the forward pass
        pred = unet(slc_tensor)
        
        # Here we compute our loss function and do the backpropagation pass
        l = loss(pred, lbl_tensor)
        l.backward()
        optimizer.step()
        
    print(f"Epoch: {epoch}, training loss: {l}")
        
# Here's a neat trick: let's visualize our last network prediction with default colormap in matplotlib:
# (note the .cpu().detach() calls - we need to move our data to CPU before manipulating it and we need to 
# stop collecting the computation graph)

plt.imshow(pred.cpu().detach()[0,1])
plt.savefig("pred")
plt.close()

# Let's run inference on just one slice first

# Switch model to the eval mode so that no gradient collection happens
unet.eval()

# TASK: pick a slice from the loaded training_volume Numpy array, convert it into PyTorch tensor,
# and run an inference on it, convert result into 2D NumPy array and visualize it. 
# Don't forget to normalize your data before running inference! Also keep in mind
# that our CNN return 2 channels - one for each class - target (spleen) and background 

# <YOUR CODE HERE>
# ___SOLUTION

def inference(img):
    tsr_test = torch.from_numpy(img.astype(np.single)/np.max(img)).unsqueeze(0).unsqueeze(0)
    pred = unet(tsr_test.to(device))
    return np.squeeze(pred.cpu().detach())

level = 11

img_test = training_volume[:,:,level]
pred = inference(img_test)

plt.imshow(pred[1])
plt.savefig("inference")
plt.close()

PATH = '/content/models/unet.pt'
torch.save(unet, PATH)
# ___SOLUTION



