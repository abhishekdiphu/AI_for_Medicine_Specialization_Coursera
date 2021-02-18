## Dataset :

- we'll be examining data from an RCT, measuring the effect of a particular drug combination on colon cancer. Specifically, we'll be looking the effect of Levamisole and Fluorouracil on patients who have had surgery to remove their colon cancer. After surgery, the curability of the patient depends on the remaining residual cancer. In this study, it was found that this particular drug combination had a clear beneficial effect, when compared with Chemotherapy.

- sex (binary): 1 if Male, 0 otherwise
- age (int): age of patient at start of the study
- obstruct (binary): obstruction of colon by tumor
- perfor (binary): perforation of colon
- adhere (binary): adherence to nearby organs
- nodes (int): number of lymphnodes with detectable cancer
- node4 (binary): more than 4 positive lymph nodes
- outcome (binary): 1 if died within 5 years
- TRTMT (binary): treated with levamisole + fluoroucil
- differ (one-hot): differentiation of tumor
- extent (one-hot): extent of local spread


- pay attention to the TRTMT and outcome columns. Our primary endpoint for our analysis will be the 5-year survival rate, which is captured in the outcome variable.

## Some statistics :

### Proportion Treated:

ptreatment=ntreatment/n is 49.99 %

- ntreatment  is the number of patients where TRTMT = True
- n  is the total number of patients.


## The probability of dying for patients who received the treatment is:

p(treatment, death)=n(treatment,death)/ ntreatment  = 37 %

- n(treatment,death)  is the number of patients who received the treatment and died.
- ntreatment  is the number of patients who received treatment.

##The probability of dying for patients in the control group (who did not received treatment) is:

p(control, death)=n(control,death)/(ncontrol) = 48 %

- n(control,death)  is the number of patients in the control group (did not receive the treatment) who died.
- ncontrol  is the number of patients in the control group (did not receive treatment).



















# Absolute risk reduction:
- Let's say a doctor is trying out a new treatment and wants to see how effective the treatment is going to be for reducing heart attack 

- the doctor try out the treatment on a group of patients and not use the treatment on another group of patients.

-  The group of patients on which the doctor uses the treatment is called the treatment arm, and the group on which the doctor does not use the treatment is called the control, giving  them a placebo, which is a fake treatment that seems like a real treatment or the standard of care,

- So now you would follow these patients for a year and see what fraction of the patients in either group get a heart attack. 

- Let's say after one year you find that 2% get a heart attack in the treatment group, while 5% of the patients in the control group get a heart attack.

-  So we can now express the effect of the treatment.

- Expressing the absolute risk of a heart attack in the treatment and control groups. 

- So in the treatment group, because 2% had a heart attack, the absolute risk is going to be 0.02. 

- Similarly in the control group, because 5% had a heart attack, the absolute risk is going to be 0.05. 

-  So we take 0.05 minus 0.02, which is going to be 0.03, which we call the absolute risk reduction of the treatment which  quantify the difference in risks within with that the treatment arm.


# Randomized control trials (RCT):

- The setup of a medical experiment where we randomly allocate subjects to two or more groups treat them differently and then compare them with respect to a measured response is called a randomized control trial, or RCT. 

- so that biased results are avoided.

- it is also common to calculate p-value to find the significance of the treamtmet effect.

- if effect is = 0.003 ARR , p-value < 0.0001, there is 0.1% prob that we will observe a differnce . it is afunction of no of paitents in a group.


# NNT(Num needed to treat):
- NNT = 1/(ARR)
- if NNT = 33 % , it means , by treating 33% of paitents , we can save 1 paitent.



# Average Treatment Effect :

#### Neyman Rubin causal: 

- Expectation[yi(1) - yi(0)] = Expectation[yi(1)] - Expectation[yi(0)] , where i is the patient number , expecation is just the mean(average).

- the average treatment effect is the expectation of the difference in the potential outcomes, and the expectation can be given by taking the average. 

- benifit = -1 | harm = 1 | no effect = 0

## c- benifit : (same as c-index or harell c index)

 = (concodeance-pairs + 0.5 * risk_ties) / (permissible pairs)

## T-Learner :
- Uses two models for estimating for one model for treatment arm and one model for controlled arm.

- thats why known as two tree method.

- but can only use half the data for each of the model.


## S-Learner :
- Uses single model for estimating  for treatment arm and for controlled arm.

- thats why known as single tree method.

- but can use full of  the data for  the model.




