# Chest X-Ray Medical Diagnosis with Deep Learning

- Pre-process and prepare a real-world X-ray dataset
- Use transfer learning to retrain a DenseNet model for X-ray image classification
- Learn a technique to handle class imbalance
- Measure diagnostic performance by computing the AUC (Area Under the Curve) for the ROC - (Receiver Operating Characteristic) curve
- Visualize model activity using GradCAMs


## 1.Pre-Processing : 


## 2.Model training :

### loss function
- weighted binary cross entropy.


## 3.Evaluation tools :


### Accuacy
(true positives+true negatives)/(true positives+true negatives+false positives+false negatives) 

### ROC :
### AUC
### PR
presicion recall curve.

### Sensitivity
P(predicted_positive | actual_positive)

TP/(TP + FN)

- Sensitivity only considers output on people in the positive class

### Specificity (recall)
P(predicted_negative | actual_negative)

TN/(TN + FP)

-  specificity only considers output on people in the negative class.

### conditional probability

bayes rule:
P(A|B) = [P(B|A)* P(A)] / [P(B)]

### Sensitivity, Specificity and Prevalence

### PPV (Precision)
P(actual_postive | predicted_postive) 
or 
by bayes rules ,

= P(predicted_postive | actual_postive )*P(actual_postive)/P(predicted_positive)

sensitivity * prevalence /(sensitivity*prevelence + (1- sensitivity)*(1- prevelence))) 

- Positive predictive value (PPV) is the probability that subjects with a positive screening test truly have the disease.

TP/ TP + FP

### NPV

- Negative predictive value (NPV) is the probability that subjects with a negative screening test truly don't have the disease.
 
P(actual_neg | predicted_neg)
or 
by bayes rule ,

= P(predicted_neg | actual_neg) *P(actual_neg)/P(predicted_neg)

TN/(TN + FN)

### Confusion matrix
TP FN
FP TN

true positive (TP): The model classifies the example as positive, and the actual label also positive.
false positive (FP): The model classifies the example as positive, but the actual label is negative.
true negative (TN): The model classifies the example as negative, and the actual label is also negative.
false negative (FN): The model classifies the example as negative, but the label is actually positive.

To compute binary class predictions, we need to convert these to either 0 or 1.
We'll do this using a threshold value  th .
Any model outputs above  th  are set to 1, and below  th  are set to 0.

### Calculating PPV in terms of sensitivity, specificity

### Prevalence

In a medical context, prevalence is the proportion of people in the population who have the disease (or condition, etc).
In machine learning terms, this is the proportion of positive examples. The expression for prevalence is:

prevalence=1N∑iyi




## 4.Explainable-AI (model explanation)
#### Gradcam