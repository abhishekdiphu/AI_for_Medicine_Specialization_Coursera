import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as cindex
from sklearn.model_selection import train_test_split



def harrell_c(y_true, scores, event):
    '''
    Compute Harrel C-index given true event/censoring times,
    model output, and event indicators.
    
    Args:
        y_true (array): array of true event times
        scores (array): model risk scores
        event (array): indicator, 1 if event occurred at that index, 0 for censorship
    Returns:
        result (float): C-index metric
    '''
    
    n = len(y_true)
    assert (len(scores) == n and len(event) == n)
    
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    
    result = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' and 'pass' with your code) ###
    
    # use double for loop to go through cases
    for i in range(n):
        # set lower bound on j to avoid double counting
        for j in range(i+1, n):
            
            # check if at most one is censored
            if event[i] == 1  or  event[j] == 1 :
                pass
            
                # check if neither are censored
                if event[i] == 1 and event[j] == 1:
                    permissible += 1
                    
                    # check if scores are tied
                    if scores[i] == scores[j]:
                        ties +=1
                    
                    # check for concordant
                    elif y_true[i] > y_true[j] and scores[i] < scores[j] :
                        concordant +=1
                    elif y_true[i] < y_true[j] and scores[i] > scores[j]:
                        concordant +=1
                
                # check if one is censored
                elif event[i] != event[j]:
    
                    # get censored index
                    censored = j
                    uncensored = i
                    
                    if event[i] == 0:
                        censored = i
                        uncensored = j
                        
                    # check if permissible
                    # Note: in this case, we are assuming that censored at a time
                    # means that you did NOT die at that time. That is, if you
                    # live until time 30 and have event = 0, then you lived THROUGH
                    # time 30.
                    if y_true[censored] >= y_true[uncensored]:
                        permissible +=1
                        
                        # check if scores are tied
                        if scores[censored] == scores[uncensored]:
                            # update ties 
                            ties +=1
                            
                        # check if scores are concordant 
                        if scores[uncensored] > scores[censored]:
                            concordant +=1
    
    # set result to c-index computed from number of concordant pairs,
    # number of ties, and number of permissible pairs (REPLACE 0 with your code)  
    
    print(concordant)
    print(ties)
    print(permissible)
    result = (concordant  + 0.5 * ties)/permissible
    
    ### END CODE HERE ###
    
    return result   




    # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def hazard_ratio(case_1, case_2, cox_params):
    '''
    Return the hazard ratio of case_1 : case_2 using
    the coefficients of the cox model.
    
    Args:
        case_1 (np.array): (1 x d) array of covariates
        case_2 (np.array): (1 x d) array of covariates
        model (np.array): (1 x d) array of cox model coefficients
    Returns:
        hazard_ratio (float): hazard ratio of case_1 : case_2
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    hr =  np.exp(np.dot(cox_params,case_1.T)) / np.exp(np.dot(cox_params,case_2.T))
    
    ### END CODE HERE ###
    
    return hr





# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def to_one_hot(dataframe, columns):
    '''
    Convert columns in dataframe to one-hot encoding.
    Args:
        dataframe (dataframe): pandas dataframe containing covariates
        columns (list of strings): list categorical column names to one hot encode
    Returns:
        one_hot_df (dataframe): dataframe with categorical columns encoded
                            as binary variables
    '''
    one_hot_df =pd.get_dummies(data=dataframe,columns=columns , dtype=np.float64, drop_first =True)
    
    return one_hot_df