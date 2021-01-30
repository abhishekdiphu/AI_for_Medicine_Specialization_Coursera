# Chest X-Ray Medical Diagnosis with Deep Learning

- Pre-process and prepare a real-world X-ray dataset
- Use transfer learning to retrain a DenseNet model for X-ray image classification
- Learn a technique to handle class imbalance
- Measure diagnostic performance by computing the AUC (Area Under the Curve) for the ROC - (Receiver Operating Characteristic) curve
- Visualize model activity using GradCAMs


Evaluation
AUC and ROC curves PR curve and classification report.

# 1.Pre-Processing 

# 2.Model training

# 3.Evaluation
## ROC
## AUC
## PR

## Sensitivity

## Specificity

## conditional probability

## Sensitivity, Specificity and Prevalence

## PPV, NPV

Rewriting PPV
PPV = P(pos | \hat{pos})PPV=P(pos∣ pos^).  
(pospos is "actually positive" and \hat{pos} pos^ is "predicted positive").

By Bayes rule, this is, 
PPV = \frac{P(\hat{pos} | pos) \times P(pos)}{P(\hat{pos})}PPV= P(pos^)P(pos^∣pos)×P(pos)
​	

## Confusion matrix

## Calculating PPV in terms of sensitivity, specificity and prevalence


# 4.Explainable-AI (model explanation)

## Gradcam