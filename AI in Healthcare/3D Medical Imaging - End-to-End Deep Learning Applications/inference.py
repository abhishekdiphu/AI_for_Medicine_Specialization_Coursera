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


np.unique(training_label)

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print(device)


PATH = '/content/models/unet.pt' 
unet = torch.load(PATH)
unet.to(device)
unet.eval()

def inference(img):
    tsr_test = torch.from_numpy(img.astype(np.single)/np.max(img)).unsqueeze(0).unsqueeze(0)
    pred = unet(tsr_test.to(device))
    return np.squeeze(pred.cpu().detach())

level = 15
for i in range(0,level):
    img_test = training_volume[:,:,i]
    pred = inference(img_test)
    plt.imshow(pred[1])
    plt.savefig('/content/results/pred'+str(i))
    plt.close()

# ___SOLUTION

# Now let's convert this into binary mask using PyTorch's argmax function:

mask = torch.argmax(pred, dim=0)
plt.imshow(mask)


# TASK: Now you have all you need to create a full NIFTI volume. Compute segmentation predictions for each slice of your volume
# and turn them into NumPy array of the same shape as the original volume

# <YOUR CODE HERE>
# ___SOLUTION
mask3d = np.zeros(training_volume.shape)

for slc_ix in range(training_volume.shape[2]):
    pred = inference(training_volume[:,:,slc_ix])
    mask3d[:,:,slc_ix] = torch.argmax(pred, dim=0)

# Finally, save our NIFTI image, copying affine from the original image

org_volume = nib.load("data/spleen1_img.nii.gz")
print("affine :" , org_volume.affine)
img_out = nib.Nifti1Image(mask3d, org_volume.affine)
nib.save(img_out, "data/out.nii.gz")





