# Chest X-Ray tuberculosis Medical Diagnosis with Deep Learning

- Pre-process and prepare a real-world X-ray dataset
- Use transfer learning to retrain a DenseNet model for X-ray image  binary-classification
- Learn a technique to handle class imbalance
- Measure diagnostic performance by computing the AUC (Area Under the Curve) for the ROC - (Receiver Operating Characteristic) curve
- Visualize model activity using GradCAMs




## 1.Pre-Processing : 

- resized to (320, 320)
- noramlized with mean of 0 and varience of 1.
- The maximum pixel value is 1.5113 and the minimum is -2.3374
- The mean value of the pixels is -0.0000 and the standard deviation is 1.0000

<img src="extra/STANDERDIZE.png" width="300px"/>      <img src="extra/STANDERDIZE_IMG.png" width="300px"/>


### Datasets
Shenzhen Hospital X-ray Set:X-ray images in this data set have been collected by Shenzhen No.3 Hospital in Shenzhen, Guangdong providence,China. The x-rays were acquired as part of the routine care at Shenzhen Hospital. The set contains images in JPEG format. There are 326 normal x-raysand 336 abnormal x-rays showing various manifestations of tuberculosis.


##### training set:
- The class normal has 249 samples
- The class tuberculosis has 281 samples
##### Notes :
- Total 662  2D x-ray images .
- The dimensions of the image are 2939 pixels width and 2985 pixels height, one single color channel
- The maximum pixel value is 1.0000 and the minimum is 0.0000
- The mean value of the pixels is 0.6082 and the standard deviation is 0.2588
- Link to the dataset :
https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html
For additional information about these datasets, please refer to the paper : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

<img src="extra/XRAY.png" width="300px"/>     <img src="extra/PIXEL_INTENSITY.png" width="300px"/>


## 2.Model training :
#### DenseNet architecture :

<img src="extra/densenet.png" width="400px"/>

#### loss function

Binary cross entropy loss

#### training loss :
<img src="images/training_loss.png" width="400px"/>




## 3.Evaluation tools :


#### Accuacy
(true positives+true negatives)/(true positives+true negatives+false positives+false negatives) 

#### ROC :
<img src="extra/AUC.png" width="400px"/> <img src="extra/PRCURVE.png" width="400px"/> <img src="extra/CI.png" width="400px"/> 



Prediction of 0.5 and above should be treated as positive and otherwise it should be treated as negative. This however was a rather arbitrary choice. One way to see this, is to look at a very informative visualization called the receiver operating characteristic (ROC) curve.

The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The ideal point is at the top left, with a true positive rate of 1 and a false positive rate of 0. The various points on the curve are generated by gradually changing the threshold.

The area under the ROC curve is also called AUCROC or C-statistic and is a measure of goodness of fit. In medical literature this number also gives the probability that a randomly selected patient who experienced a condition had a higher risk score than a patient who had not experienced the event. This summarizes the model output across all thresholds, and provides a good sense of the discriminative power of a given model.


#### Sensitivity:
- P(predicted_positive | actual_positive)
- TP/(TP + FN)
- Sensitivity only considers output on people in the positive class

#### Specificity (recall):

- P(predicted_negative | actual_negative)
- TN/(TN + FP)
-  specificity only considers output on people in the negative class.

#### conditional probability:

bayes rule:
P(A|B) = [P(B|A)* P(A)] / [P(B)]

#### PPV (Precision):
- P(actual_postive | predicted_postive) 
or 
by bayes rules ,

= P(predicted_postive | actual_postive )*P(actual_postive)/P(predicted_positive)
- Positive predictive value (PPV) is the probability that subjects with a positive screening test truly have the disease.
- TP/ TP + FP

#### NPV:

- Negative predictive value (NPV) is the probability that subjects with a negative screening test truly don't have the disease.
- P(actual_neg | predicted_neg)
or 
by bayes rule ,

= P(predicted_neg | actual_neg) *P(actual_neg)/P(predicted_neg)
- TN/(TN + FN)

#### Confusion matrix:
<img src="extra/CONFUSION_MATRIX.png" width="400px"/>

TP FN
FP TN

true positive (TP): The model classifies the example as positive, and the actual label also positive.
false positive (FP): The model classifies the example as positive, but the actual label is negative.
true negative (TN): The model classifies the example as negative, and the actual label is also negative.
false negative (FN): The model classifies the example as negative, but the label is actually positive.

- To compute binary class predictions, we need to convert these to either 0 or 1.
We'll do this using a threshold value  th .
- Any model outputs above  th  are set to 1, and below  th  are set to 0.

#### Calculating PPV in terms of sensitivity, specificity:
- sensitivity * prevalence /(sensitivity*prevelence + (1- sensitivity)*(1- prevelence))) 

#### Prevalence

In a medical context, prevalence is the proportion of people in the population who have the disease (or condition, etc).
In machine learning terms, this is the proportion of positive examples. The expression for prevalence is:

prevalence=1N∑iyi

#### Classification Report:
<img src="extra/CLS_RPT.png" width="400px"/>


#### Evaluation report :

<img src="extra/TABLE.png" width="600px"/>

## 4.Explainable-AI (model explanation)

#### Gradcam:
<img src="extra/CHNCXR_0600_1.png" width="800px"/> 


- they belong to the class of class-activation maps. they are called , gradient-class-activation map.
- usually used for interpreting which part of the features in an image does it contributes to the predicted class of the image.
- Since early layers deals with low level features , and last layers deals with high level features , so the last convolutional layer is being taken for this purpose.(knwn as spatial maps).
- these maps are usually smaller than the input images , between 7x7xk to 14x14xk , depending on the architectures of CNN.
- now , if there are k activation maps A1 TO Ak , then , first calulate average weight for all the activation maps , by taking partitial derivates , of output score 'y' for a particular class with respect to each feature Aij of the activation map A.
- with the help of these , activation map weights ,  we calculate the localization map from these activation maps (A1, A2.....AK).
- We apply the relu(0,max) on the weighted sum of the all k activation maps(A) to get only the postive influence and discard the negative incluence to obtain the the localization maps.
- trasnlate the localization map to heatmap  by a colormap . it is the same shap as the activation map.
- then resize  the heatmap to the size  of  the input image though interpolation technique.
- overlay the heatmap onto  the  input image with some transparency.


#### Integrated Gradients:
<img src="images/integrated_grad.png" width="800px"/>

[Integrated Gradients](https://arxiv.org/abs/1703.01365) is a technique for
attributing a classification model's prediction to its input features. It is
a model interpretability technique: you can use it to visualize the relationship
between input features and model predictions.

Integrated Gradients is a variation on computing
the gradient of the prediction output with regard to features of the input.
To compute integrated gradients, we need to perform the following steps:

1. Identify the input and the output. In our case, the input is an image and the
output is the last layer of our model (dense layer with softmax activation).

2. Compute which features are important to a neural network
when making a prediction on a particular data point. To identify these features, we
need to choose a baseline input. A baseline input can be a black image (all pixel
values set to zero) or random noise. The shape of the baseline input needs to be
the same as our input image, e.g. (299, 299, 3).

3. Interpolate the baseline for a given number of steps. The number of steps represents
the steps we need in the gradient approximation for a given input image. The number of
steps is a hyperparameter. The authors recommend using anywhere between
20 and 1000 steps.

4. Preprocess these interpolated images and do a forward pass.
5. Get the gradients for these interpolated images.
6. Approximate the gradients integral using the trapezoidal rule.

To read in-depth about integrated gradients and why this method works,
consider reading this excellent
[article](https://distill.pub/2020/attribution-baselines/).


## Run 
- pip install -r requirement.txt
- download the dataset and unzip and put it in nih/ folder.
- **python training.py** in your conda or virtualenv environment.
- training labels are nih/train.csv and test label are nih/test.csv ,
- One can also perform cross validation with the datasets label file , nih/dataset_03.csv file.

**References:**

- Integrated Gradients original [paper](https://arxiv.org/abs/1703.01365)
- [Original implementation](https://github.com/ankurtaly/Integrated-Gradients)
- https://radiopaedia.org/





