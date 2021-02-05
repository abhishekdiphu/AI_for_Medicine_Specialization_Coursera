

## Diabetic Retinopathy
Retinopathy is an eye condition that causes changes to the blood vessels in the part of the eye called the retina. This often leads to vision changes or blindness. Diabetic patients are known to be at high risk for retinopathy.

## Logistic Regression
Logistic regression is an appropriate analysis to use for predicting the probability of a binary outcome. In our case, this would be the probability of having or not having diabetic retinopathy. Logistic Regression is one of the most commonly used algorithms for binary classification. It is used to find the best fitting model to describe the relationship between a set of features (also referred to as input, independent, predictor, or explanatory variables) and a binary outcome label (also referred to as an output, dependent, or response variable). Logistic regression has the property that the output prediction is always in the range  [0,1] . Sometimes this output is used to represent a probability from 0%-100%, but for straight binary classification, the output is converted to either  0  or  1  depending on whether it is below or above a certain threshold, usually  0.5 .



## 4. Mean-Normalize the Data
Let's now transform our data so that the distributions are closer to standard normal distributions.

First we will remove some of the skew from the distribution by using the log transformation. Then we will "standardize" the distribution so that it has a mean of zero and standard deviation of 1. Recall that a standard normal distribution has mean of zero and standard deviation of 1.



## Evaluate the Model Using the C-index
Now that we have a model, we need to evaluate it. We'll do this using the c-index.

The c-index measures the discriminatory power of a risk score.
Intuitively, a higher c-index indicates that the model's prediction is in agreement with the actual outcomes of a pair of patients.
The formula for the c-index is


- cindex=concordant+0.5×tiespermissible
 
- A permissible pair is a pair of patients who have different outcomes.
- A concordant pair is a permissible pair in which the patient with the higher risk score also has the worse outcome.
- A tie is a permissible pair where the patients have the same risk score.


