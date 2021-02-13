

# 4. Mean-Normalize the Data
Let's now transform our data so that the distributions are closer to standard normal distributions.

First we will remove some of the skew from the distribution by using the log transformation. Then we will "standardize" the distribution so that it has a mean of zero and standard deviation of 1. Recall that a standard normal distribution has mean of zero and standard deviation of 1.

# Evaluate the Model Using the C-index
Now that we have a model, we need to evaluate it. We'll do this using the c-index.

The c-index measures the discriminatory power of a risk score.
Intuitively, a higher c-index indicates that the model's prediction is in agreement with the actual outcomes of a pair of patients.
The formula for the c-index is
* cindex=concordant+0.5×tiespermissible
 
- A permissible pair is a pair of patients who have different outcomes.
- A concordant pair is a permissible pair in which the patient with the higher risk score also - has the worse outcome.
- A tie is a permissible pair where the patients have the same risk score.










# Cox Proportional Hazards and Random Survival Forests
develop risk models using survival data and a combination of linear and non-linear techniques. We'll be using a dataset with survival data of patients with Primary Biliary Cirrhosis (pbc). PBC is a progressive disease of the liver caused by a buildup of bile within the liver (cholestasis) that results in damage to the small bile ducts that drain bile from the liver. Our goal will be to understand the effects of different factors on the survival times of the patients. Along the way you'll learn about the following topics:

- Cox Proportional Hazards
- Data Preprocessing for Cox Models.
- Random Survival Forests
- Permutation Methods for Interpretation.