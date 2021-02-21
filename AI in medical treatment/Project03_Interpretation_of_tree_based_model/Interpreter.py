import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import sklearn
import lifelines
import shap

import os
from util import *


viz_folder = "/content/visualization/shap"

if os.path.exists(viz_folder):
  print("visulaization folder exist")
else:
  print("visulaization folder does not exist...making new folder")
  os.makedirs(viz_folder)


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def permute_feature(df, feature):
    """
    Given dataset, returns version with the values of
    the given feature randomly permuted. 

    Args:
        df (dataframe): The dataset, shape (num subjects, num features)
        feature (string): Name of feature to permute
    Returns:
        permuted_df (dataframe): Exactly the same as df except the values
                                of the given feature are randomly permuted.
    """
    permuted_df = df.copy(deep=True) # Make copy so we don't change original df

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Permute the values of the column 'feature'
    permuted_features = np.random.permutation(permuted_df[feature])
    
    # Set the column 'feature' to its permuted values.
    permuted_df[feature] = permuted_features
    
    ### END CODE HERE ###

    return permuted_df



# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def permutation_importance(X, y, model, metric, num_samples = 100):
    """
    Compute permutation importance for each feature.

    Args:
        X (dataframe): Dataframe for test data, shape (num subject, num features)
        y (np.array): Labels for each row of X, shape (num subjects,)
        model (object): Model to compute importances for, guaranteed to have
                        a 'predict_proba' method to compute probabilistic 
                        predictions given input
        metric (function): Metric to be used for feature importance. Takes in ground
                           truth and predictions as the only two arguments
        num_samples (int): Number of samples to average over when computing change in
                           performance for each feature
    Returns:
        importances (dataframe): Dataframe containing feature importance for each
                                 column of df with shape (1, num_features)
    """

    importances = pd.DataFrame(index = ['importance'], columns = X.columns)
    
    # Get baseline performance (note, you'll use this metric function again later)
    baseline_performance = metric(y, model.predict_proba(X)[:, 1])

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Iterate over features (the columns in the importances dataframe)
    for feature in importances.columns: # complete this line
        
        # Compute 'num_sample' performances by permutating that feature
        
        # You'll see how the model performs when the feature is permuted
        # You'll do this num_samples number of times, and save the performance each time
        # To store the feature performance,
        # create a numpy array of size num_samples, initialized to all zeros
        feature_performance_arr = np.zeros(num_samples)
        
        # Loop through each sample
        for i in range(num_samples): # complete this line
            
            # permute the column of dataframe X
            perm_X = permute_feature(X, feature)
            
            # calculate the performance with the permuted data
            # Use the same metric function that was used earlier
            feature_performance_arr[i] =  metric(y, model.predict_proba(perm_X)[:, 1])
    
    
        # Compute importance: absolute difference between 
        # the baseline performance and the average across the feature performance
        importances[feature]['importance'] = np.abs(baseline_performance - np.mean(feature_performance_arr))
        
    ### END CODE HERE ###

    return importances

# This sets a common size for all the figures we will draw.
plt.rcParams['figure.figsize'] = [10, 7]





rf = pickle.load(open('nhanes_rf.sav', 'rb')) # Loading the model
test_df = pd.read_csv('nhanest_test.csv')
test_df = test_df.drop(test_df.columns[0], axis=1)
X_test = test_df.drop('y', axis=1)
y_test = test_df.loc[:, 'y']
cindex_test = cindex(y_test, rf.predict_proba(X_test)[:, 1])

print("Model C-index on test: {}".format(cindex_test))


X_test_risky = X_test.copy(deep=True)
X_test_risky.loc[:, 'risk'] = rf.predict_proba(X_test)[:, 1] # Predicting our risk.
X_test_risky = X_test_risky.sort_values(by='risk', ascending=False) # Sorting by risk value.
print(X_test_risky.head())


importances = permutation_importance(X_test, y_test, rf, cindex, num_samples=100)
print(importances)

importances.T.plot.bar()
plt.ylabel("Importance")
l = plt.legend()
l.remove()
plt.savefig('importance')
plt.show()
plt.close()


explainer = shap.TreeExplainer(rf)

for i in range(0 ,20 ,2):
    i = i # Picking an individual
    shap_value = explainer.shap_values(X_test.loc[X_test_risky.index[i], :])[1]
    shap.force_plot(explainer.expected_value[1], shap_value, feature_names=X_test.columns, matplotlib=True , show=False)
    plt.savefig(str(i))

shap_values = shap.TreeExplainer(rf).shap_values(X_test)[1]

shap.summary_plot(shap_values, X_test, show = False )
plt.savefig(viz_folder + '/summary_plot')
