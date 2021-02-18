import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random
import lifelines
import itertools
from sklearn.linear_model import LogisticRegression
from lifelines.utils import concordance_index

plt.rcParams['figure.figsize'] = [10, 7]



# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def proportion_treated(df):
    """
    Compute proportion of trial participants who have been treated

    Args:
        df (dataframe): dataframe containing trial results. Column
                      'TRTMT' is 1 if patient was treated, 0 otherwise.
  
    Returns:
        result (float): proportion of patients who were treated
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    proportion = len(df[df['TRTMT'] == 1])/len(df.TRTMT)
    ### END CODE HERE ###

    return proportion





def event_rate(df):
    '''
    Compute empirical rate of death within 5 years
    for treated and untreated groups.

    Args:
        df (dataframe): dataframe containing trial results. 
                          'TRTMT' column is 1 if patient was treated, 0 otherwise. 
                            'outcome' column is 1 if patient died within 5 years, 0 otherwise.
  
    Returns:
        treated_prob (float): empirical probability of death given treatment
        untreated_prob (float): empirical probability of death given control
    '''
    
    treated_prob = 0.0
    control_prob = 0.0
        
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    treated_prob = len(df[(df['TRTMT'] == 1) & (df['outcome'] == 1)]) / len(df[df['TRTMT'] == 1])
    control_prob = len(df[(df['TRTMT'] == 0) & (df['outcome'] == 1)]) / len(df[df['TRTMT'] == 0])
    
    ### END CODE HERE ###

    return treated_prob, control_prob



# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_treatment_effect(lr, data):
    theta_TRTMT = 0.0
    TRTMT_OR = 0.0
    coeffs = {data.columns[i]:lr.coef_[0][i] for i in range(len(data.columns))}
    print(coeffs)
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # get the treatment coefficient
    theta_TRTMT = coeffs['TRTMT']
    
    # calculate the Odds ratio for treatment
    TRTMT_OR = np.exp(theta_TRTMT)
    
    ### END CODE HERE ###
    return theta_TRTMT, TRTMT_OR



# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def OR_to_ARR(p, OR):
    """
    Compute ARR for treatment for individuals given
    baseline risk and odds ratio of treatment.

    Args:
        p (float): baseline probability of risk (without treatment)
        OR (float): odds ratio of treatment versus baseline

    Returns:
        ARR (float): absolute risk reduction for treatment 
      """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # compute baseline odds from p
    odds_baseline = p/(1 -p)

    # compute odds of treatment using odds ratio
    odds_trtmt = odds_baseline * OR

    # compute new probability of death from treatment odds
    p_trtmt = odds_trtmt /(1 + odds_trtmt)

    # compute ARR using treated probability and baseline probability 
    ARR = p - p_trtmt
    
    ### END CODE HERE ###
    
    return ARR



# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def base_risks(X, lr_model):
    """
    Compute baseline risks for each individual in X.

    Args:
        X (dataframe): data from trial. 'TRTMT' column
                       is 1 if subject retrieved treatment, 0 otherwise
        lr_model (model): logistic regression model
    
    Returns:
        risks (np.array): array of predicted baseline risk
                          for each subject in X
    """
    
    # first make a copy of the dataframe so as not to overwrite the original
    X = X.copy(deep=True)
    
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Set the treatment variable to assume that the patient did not receive treatment
    
    X['TRTMT'] = X['TRTMT'].replace(True ,0) 
    
    
    # Input the features into the model, and predict the probability of death.
    risks = np.array([])
    for i in range(X.shape[0]):
        risks = np.append(risks,lr_model.predict_proba(X)[i][1])
    
    
    
    # END CODE HERE

    return risks






# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def lr_ARR_quantile(X, y, lr):
    
    # first make a deep copy of the features dataframe to calculate the base risks
    X = X.copy(deep=True)
    
    # Make another deep copy of the features dataframe to store baseline risk, risk_group, and y
    df = X.copy(deep=True)

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Calculate the baseline risks (use the function that you just implemented)
    baseline_risk = base_risks(X, lr)
    print(baseline_risk)
    
    # bin patients into 10 risk groups based on their baseline risks
    risk_groups = pd.cut(baseline_risk , 10)
        
    # Store the baseline risk, risk_groups, and y into the new dataframe
    df.loc[:, 'baseline_risk'] = baseline_risk
    df.loc[:, 'risk_group'] = risk_groups
    df.loc[:, 'y'] = y

    # select the subset of patients who did not actually receive treatment
    df_baseline = df[df['TRTMT'] == 0]
    
    # select the subset of patients who did actually receive treatment
    df_treatment = df[df['TRTMT'] == 1]
    
    # For baseline patients, group them by risk group, select their outcome 'y', and take the mean
    baseline_mean_by_risk_group = df_baseline.groupby('risk_group')['y'].mean()
    
    # For treatment patients, group them by risk group, select their outcome 'y', and take the mean
    treatment_mean_by_risk_group = df_treatment.groupby('risk_group')['y'].mean()
    
    # Calculate the absolute risk reduction by risk group (baseline minus treatment)
    arr_by_risk_group = baseline_mean_by_risk_group -  treatment_mean_by_risk_group
    
    # Set the index of the arr_by_risk_group dataframe to the average baseline risk of each risk group 
    # Use data for all patients to calculate the average baseline risk, grouped by risk group.
    arr_by_risk_group.index = df.groupby('risk_group')['baseline_risk'].mean()

    ### END CODE HERE ###
    
    # Set the name of the Series to 'ARR'
    arr_by_risk_group.name = 'ARR'
    

    return arr_by_risk_group




# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def c_for_benefit_score(pairs):
    """
    Compute c-statistic-for-benefit given list of
    individuals matched across treatment and control arms. 

    Args:
        pairs (list of tuples): each element of the list is a tuple of individuals,
                                the first from the control arm and the second from
                                the treatment arm. Each individual 
                                p = (pred_outcome, actual_outcome) is a tuple of
                                their predicted outcome and actual outcome.
    Result:
        cstat (float): c-statistic-for-benefit computed from pairs.
    """
    
    # mapping pair outcomes to benefit
    obs_benefit_dict = {
        (0, 0): 0,
        (0, 1): -1,
        (1, 0): 1,
        (1, 1): 0,
    }
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None', 'False', and 'pass' with your code) ###

    # compute observed benefit for each pair
    obs_benefit = [obs_benefit_dict[(p[0][1] , p[1][1])] for p in pairs]

    # compute average predicted benefit for each pair
    pred_benefit = [ np.mean([p[0][0], p[1][0]]) for p in pairs]

    concordant_count, permissible_count, risk_tie_count = 0, 0, 0

    # iterate over pairs of pairs
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            
            # if the observed benefit is different, increment permissible count
            if obs_benefit[i] != obs_benefit[j]:

                # increment count of permissible pairs
                permissible_count += 1 
                
                # if concordant, increment count
                
                if (obs_benefit[i] < obs_benefit[j]) == (pred_benefit[i] < pred_benefit[j]): # change to check for concordance
                    
                    concordant_count += 1

                # if risk tie, increment count
                if (pred_benefit[i] == pred_benefit[j]): #change to check for risk ties
                    
                    risk_tie_count += 1


    # compute c-statistic-for-benefit
    cstat = (concordant_count + 0.5 * risk_tie_count) / permissible_count
    
    # END CODE HERE
    
    return cstat







# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def c_statistic(pred_rr, y, w, random_seed=0):
    """
    Return concordance-for-benefit, the proportion of all matched pairs with
    unequal observed benefit, in which the patient pair receiving greater
    treatment benefit was predicted to do so.

    Args: 
        pred_rr (array): array of predicted risk reductions
        y (array): array of true outcomes
        w (array): array of true treatments 
    
    Returns: 
        cstat (float): calculated c-stat-for-benefit
    """
    assert len(pred_rr) == len(w) == len(y)
    random.seed(random_seed)
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Collect pred_rr, y, and w into tuples for each patient
    tuples = tuple(zip(pred_rr , y , w))
    print(tuples)
    
    # Collect untreated patient tuples, stored as a list
    untreated = list(filter(lambda x : x[2] == 0 , tuples))
    
    # Collect treated patient tuples, stored as a list
    treated = list(filter(lambda x : x[2] == 1 , tuples))

    # randomly subsample to ensure every person is matched
    
    # if there are more untreated than treated patients,
    # randomly choose a subset of untreated patients, one for each treated patient.

    if len(treated) < len(untreated):
        untreated = random.sample(untreated, len(treated))
        
    # if there are more treated than untreated patients,
    # randomly choose a subset of treated patients, one for each treated patient.
    if len(untreated) < len(treated):
        treated = random.sample(treated, len(untreated))
        
    assert len(untreated) == len(treated)

    # Sort the untreated patients by their predicted risk reduction
    untreated = sorted(untreated, key=lambda x: x[0])
    
    # Sort the treated patients by their predicted risk reduction
    treated   = sorted(treated, key=lambda x: x[0])
    
    # match untreated and treated patients to create pairs together
    pairs = list(zip(untreated , treated))

    # calculate the c-for-benefit using these pairs (use the function that you implemented earlier)
    cstat = c_for_benefit_score(pairs)
    
    ### END CODE HERE ###
    
    return cstat




def treatment_control(X):
    """Create treatment and control versions of data"""
    X_treatment = X.copy(deep=True)
    X_control = X.copy(deep=True)
    X_treatment.loc[:, 'TRTMT'] = 1
    X_control.loc[:, 'TRTMT'] = 0
    return X_treatment, X_control

def risk_reduction(model, data_treatment, data_control):
    """Compute predicted risk reduction for each row in data"""
    treatment_risk = model.predict_proba(data_treatment)[:, 1]
    control_risk = model.predict_proba(data_control)[:, 1]
    return control_risk - treatment_risk




def quantile_benefit(X, y, arr_hat):
    df = X.copy(deep=True)
    df.loc[:, 'y'] = y
    df.loc[:, 'benefit'] = arr_hat
    benefit_groups = pd.qcut(arr_hat, 10)
    df.loc[:, 'benefit_groups'] = benefit_groups
    empirical_benefit = df.loc[df.TRTMT == 0, :].groupby('benefit_groups').y.mean() - df.loc[df.TRTMT == 1].groupby('benefit_groups').y.mean()
    avg_benefit = df.loc[df.TRTMT == 0, :].y.mean() - df.loc[df.TRTMT==1, :].y.mean()
    return empirical_benefit, avg_benefit

def plot_empirical_risk_reduction(emp_benefit, av_benefit, model,  figname = 'emprical'):
    plt.scatter(range(len(emp_benefit)), emp_benefit)
    plt.xticks(range(len(emp_benefit)), range(1, len(emp_benefit) + 1))
    plt.title("Empirical Risk Reduction vs. Predicted ({})".format(model))
    plt.ylabel("Empirical Risk Reduction")
    plt.xlabel("Predicted Risk Reduction Quantile")
    plt.plot(range(10), [av_benefit]*10, linestyle='--', label='average RR')
    plt.legend(loc='lower right')
    plt.savefig(figname)
    plt.show()
    plt.close()
