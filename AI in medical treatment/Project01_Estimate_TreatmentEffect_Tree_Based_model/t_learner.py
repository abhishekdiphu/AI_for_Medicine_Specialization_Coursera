import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random
import lifelines
import itertools
from utils import *
# As usual, split into dev and test set
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = [10, 7]



# UNQ_C9 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
class TLearner():
    """
    T-Learner class.

    Attributes:
      treatment_estimator (object): fitted model for treatment outcome
      control_estimator (object): fitted model for control outcome
    """                               
    def __init__(self, treatment_estimator, control_estimator):
        """
        Initializer for TLearner class.
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # set the treatment estimator
        self.treatment_estimator = treatment_estimator
        
        # set the control estimator 
        self.control_estimator = control_estimator
        
        ### END CODE HERE ###

    def predict(self, X):
        """
        Return predicted risk reduction for treatment for given data matrix.

        Args:
          X (dataframe): dataframe containing features for each subject
    
        Returns:
          preds (np.array): predicted risk reduction for each row of X
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # predict the risk of death using the control estimator
        risk_control = self.control_estimator.predict_proba(X)[: , 1]
        
        # predict the risk of death using the treatment estimator
        risk_treatment = self.treatment_estimator.predict_proba(X)[:, 1]
        
        # the predicted risk reduction is control risk minus the treatment risk
        pred_risk_reduction =  risk_control - risk_treatment
        
        ### END CODE HERE ###
                
        return pred_risk_reduction





# UNQ_C10 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def holdout_grid_search(clf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparam, verbose=False):
    '''
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.
    
    Input:
        clf: sklearn classifier
        X_train_hp (dataframe): dataframe for training set input variables
        y_train_hp (dataframe): dataframe for training set targets
        X_val_hp (dataframe): dataframe for validation set input variables
        y_val_hp (dataframe): dataframe for validation set targets
        hyperparam (dict): hyperparameter dictionary mapping hyperparameter
                                                names to range of values for grid search
    
    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                                                                 validation set
    '''
    # Initialize best estimator
    best_estimator = clf
    
    # initialize best hyperparam
    best_hyperparam = {}
    
    # initialize the c-index best score to zero
    best_score = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # Get the values of the hyperparam and store them as a list of lists
    hyper_param_l = list(hyperparam.values())
    
    # Generate a list of tuples with all possible combinations of the hyperparams
    combination_l_of_t = list(itertools.product(*hyper_param_l))
        
    # Initialize the list of dictionaries for all possible combinations of hyperparams
    combination_l_of_d = []
    
    # loop through each tuple in the list of tuples
    for val_tuple in combination_l_of_t: # complete this line
        param_d = {}
        
        # Enumerate each key in the original hyperparams dictionary
        for i, k in enumerate(hyperparam.keys()): # complete this line
            
            # add a key value pair to param_d for each value in val_tuple
            param_d[k] = val_tuple[i]
        
        # append the param_dict to the list of dictionaries
        combination_l_of_d.append(param_d)
        
    
    # For each hyperparam dictionary in the list of dictionaries:
    for param_d in combination_l_of_d: # complete this line
        
        # Set the model to the given hyperparams
        estimator = clf(**param_d)
        
        # Train the model on the training features and labels
        estimator.fit(X_train_hp, y_train_hp)
        
        # Predict the risk of death using the validation features
        preds = estimator.predict_proba(X_val_hp)[: , 1]
        
        # Evaluate the model's performance using the regular concordance index
        estimator_score = concordance_index(y_val_hp, preds)
        
        # if the model's c-index is better than the previous best:
        if estimator_score > best_score : # complete this line

            # save the new best score
            best_score = estimator_score
            
            # same the new best estimator
            best_estimator = estimator
            
            # save the new best hyperparams
            best_hyperparam = param_d
                
    ### END CODE HERE ###

    if verbose:
        print("hyperparam:")
        display(hyperparam)
        
        print("hyper_param_l")
        display(hyper_param_l)
        
        print("combination_l_of_t")
        display(combination_l_of_t)
        
        print(f"combination_l_of_d")
        display(combination_l_of_d)
        
        print(f"best_hyperparam")
        display(best_hyperparam)
        print(f"best_score: {best_score:.4f}")
        
    return best_estimator, best_hyperparam




# UNQ_C11 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def treatment_dataset_split(X_train, y_train, X_val, y_val):
    """
    Separate treated and control individuals in training
    and testing sets. Remember that returned
    datasets should NOT contain the 'TRMT' column!

    Args:
        X_train (dataframe): dataframe for subject in training set
        y_train (np.array): outcomes for each individual in X_train
        X_val (dataframe): dataframe for subjects in validation set
        y_val (np.array): outcomes for each individual in X_val
    
    Returns:
        X_treat_train (df): training set for treated subjects
        y_treat_train (np.array): labels for X_treat_train
        X_treat_val (df): validation set for treated subjects
        y_treat_val (np.array): labels for X_treat_val
        X_control_train (df): training set for control subjects
        y_control_train (np.array): labels for X_control_train
        X_control_val (np.array): validation set for control subjects
        y_control_val (np.array): labels for X_control_val
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # From the training set, get features of patients who received treatment
    X_treat_train = X_train[X_train['TRTMT']==1]
    
    # drop the 'TRTMT' column
    X_treat_train = X_treat_train.drop('TRTMT' , axis = 1)
    
    # From the training set, get the labels of patients who received treatment
    y_treat_train = y_train[X_train['TRTMT']==1]

    # From the validation set, get the features of patients who received treatment
    X_treat_val = X_val[X_val['TRTMT']==1] 
                        
    # Drop the 'TRTMT' column
    X_treat_val = X_treat_val.drop('TRTMT' , axis = 1)
                        
    # From the validation set, get the labels of patients who received treatment
    y_treat_val = y_val[X_val['TRTMT']==1]
                        
# --------------------------------------------------------------------------------------------
                        
    # From the training set, get the features of patients who did not received treatment
    X_control_train = X_train[X_train['TRTMT']==0]
                        
    # Drop the TRTMT column
    X_control_train = X_control_train.drop('TRTMT', axis=1)
                        
    # From the training set, get the labels of patients who did not receive treatment
    y_control_train = y_train[X_train['TRTMT']==0]
    
    # From the validation set, get the features of patients who did not receive treatment
    X_control_val = X_val[X_val['TRTMT'] == 0]
    
    # drop the 'TRTMT' column
    X_control_val = X_control_val.drop('TRTMT', axis=1)

    # From the validation set, get teh labels of patients who did not receive treatment
    y_control_val = y_val[X_val['TRTMT']==0]
    
    ### END CODE HERE ###

    return (X_treat_train, y_treat_train,
            X_treat_val, y_treat_val,
            X_control_train, y_control_train,
            X_control_val, y_control_val)




data = pd.read_csv("levamisole_data.csv", index_col=0)
print(f"Data Dimensions: {data.shape}")

p = proportion_treated(data)
print(f"Proportion Treated: {p} ~ {int(p*100)}%")


treated_prob, control_prob = event_rate(data)

print(f"Death rate for treated patients: {treated_prob:.4f} ~ {int(treated_prob*100)}%")
print(f"Death rate for untreated patients: {control_prob:.4f} ~ {int(control_prob*100)}%")


# As usual, split into dev and test set
from sklearn.model_selection import train_test_split
np.random.seed(18)
random.seed(1)

data = data.dropna(axis=0)
y = data.outcome
# notice we are dropping a column here. Now our total columns will be 1 less than before
X = data.drop('outcome', axis=1) 
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size = 0.25, 
                                                random_state=0)


print(f"dev set shape: {X_dev.shape}")
print(f"test set shape: {X_test.shape}")





# Import the random forest classifier to be used as the base learner
from sklearn.ensemble import RandomForestClassifier

# Split the dev data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_dev, 
                                                  y_dev, 
                                                  test_size = 0.25,
                                                  random_state = 0)




# get treatment and control arms of training and validation sets
(X_treat_train, y_treat_train, 
 X_treat_val, y_treat_val,
 X_control_train, y_control_train,
 X_control_val, y_control_val) = treatment_dataset_split(X_train, y_train,
                                                         X_val, y_val)

# hyperparameter grid (we'll use the same one for both arms for convenience)
# Note that we set random_state to zero
# in order to make the output consistent each time it's run.
hyperparams = {
    'n_estimators': [100, 200],
    'max_depth': [2, 5, 10, 40, None],
    'min_samples_leaf': [1, 0.1, 0.2],
    'random_state': [0]
}

# perform grid search with the treatment data to find the best model 
treatment_model, best_hyperparam_treat  = holdout_grid_search(RandomForestClassifier,
                                      X_treat_train, y_treat_train,
                                      X_treat_val, y_treat_val, hyperparams)


# perform grid search with the control data to find the best model 
control_model, best_hyperparam_ctrl = holdout_grid_search(RandomForestClassifier,
                                    X_control_train, y_control_train,
                                    X_control_val, y_control_val, hyperparams)



# Save the treatment and control models into an instance of the TLearner class
t_learner = TLearner(treatment_model, control_model)




# Use the t-learner to predict the risk reduction for patients in the validation set
rr_t_val = t_learner.predict(X_val.drop(['TRTMT'], axis=1))

print(f"X_val num of patients {X_val.shape[0]}")
print(f"rr_t_val num of patient predictions {rr_t_val.shape[0]}")



plt.hist(rr_t_val, bins='auto')
plt.title("Histogram of Predicted ARR, T-Learner, validation set")
plt.xlabel('predicted risk reduction')
plt.ylabel('count of patients')
plt.savefig('PRR_VS_COP')
plt.show()
plt.close()



empirical_benefit, avg_benefit = quantile_benefit(X_val, y_val, rr_t_val)
plot_empirical_risk_reduction(empirical_benefit, avg_benefit, 'T Learner [val set]', figname='val_empirical_risk_reduction01' )




c_for_benefit_tlearner_val_set = c_statistic(rr_t_val, y_val, X_val.TRTMT)
print(f"C-for-benefit statistic of T-learner on val set: {c_for_benefit_tlearner_val_set:.4f}")



# predict the risk reduction for each of the patients in the test set
rr_t_test = t_learner.predict(X_test.drop(['TRTMT'], axis=1))



# Plot a histogram of the predicted risk reduction
plt.hist(rr_t_test, bins='auto')
plt.title("Histogram of Predicted ARR for the T-learner on test set")
plt.xlabel("predicted risk reduction")
plt.ylabel("count of patients")
plt.savefig('PRR_VS_COP1')
plt.show()
plt.close()

# Plot the predicted versus empirical risk reduction for the test set
empirical_benefit, avg_benefit = quantile_benefit(X_test, y_test, rr_t_test)
plot_empirical_risk_reduction(empirical_benefit, avg_benefit, 'T Learner (test set)',  figname='test_empirical_risk_reduction02')
plt.savefig('empirical_risk_reduction_test')



# calculate the c-for-benefit of the t-learner on the test set
c_for_benefit_tlearner_test_set = c_statistic(rr_t_test, y_test, X_test.TRTMT)
print(f"C-for-benefit statistic on test set: {c_for_benefit_tlearner_test_set:.4f}")