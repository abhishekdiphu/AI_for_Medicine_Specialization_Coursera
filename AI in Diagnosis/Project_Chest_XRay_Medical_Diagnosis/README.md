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
In some studies, you may have to compute the Positive predictive value (PPV) from the sensitivity, specificity and prevalence.  Note that reviewing this reading will help you answer one of the quizzes at the end of this week!

Rewriting PPV
PPV = P(pos | \hat{pos})PPV=P(pos∣ 
pos
^
​	
 ).  

(pospos is "actually positive" and \hat{pos} 
pos
^
​	
  is "predicted positive").

By Bayes rule, this is 

 PPV = \frac{P(\hat{pos} | pos) \times P(pos)}{P(\hat{pos})}PPV= 
P( 
pos
^
​	
 )
P( 
pos
^
​	
 ∣pos)×P(pos)
​	
 

For the numerator:
Sensitivity = P(\hat{pos} | pos) Sensitivity=P( 
pos
^
​	
 ∣pos).  Recall that sensitivity is how well the model predicts actual positive cases as positive.

Prevalence = P(pos)Prevalence=P(pos).  Recall that prevalence is how many actual positives there are in the population.

For the denominator:
P(\hat{pos}) = TruePos + FalsePosP( 
pos
^
​	
 )=TruePos+FalsePos.  In other words, the model's positive predictions are the sum of when it correctly predicts positive and incorrectly predicts positive.

The true positives can be written in terms of sensitivity and prevalence.

TruePos = P(\hat{pos} | pos) \times P(pos)TruePos=P( 
pos
^
​	
 ∣pos)×P(pos), and you can use substitution to get 

TruePos = Sensitivity \times PrevalenceTruePos=Sensitivity×Prevalence

The false positives can also be written in terms of specificity and prevalence:

FalsePos = P(\hat{pos} | neg) \times P(neg)FalsePos=P( 
pos
^
​	
 ∣neg)×P(neg)

1 - specificity = P(\hat{pos} | neg )1−specificity=P( 
pos
^
​	
 ∣neg)

1 - prevalence = P(neg)1−prevalence=P(neg)

PPV rewritten:
If you substitute these into the PPV equation, you'll get

PPV = \frac{sensitivity \times prevalence}{sensitivity \times prevalence + (1 - specificity) \times (1 - prevalence)} PPV= 
sensitivity×prevalence+(1−specificity)×(1−prevalence)
sensitivity×prevalence
​	


## Confusion matrix

## Calculating PPV in terms of sensitivity, specificity and prevalence


# 4.Explainable-AI (model explanation)

## Gradcam