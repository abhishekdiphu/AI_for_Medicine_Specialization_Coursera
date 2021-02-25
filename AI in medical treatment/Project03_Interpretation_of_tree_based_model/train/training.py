#import shap
import os
import sklearn
import itertools
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# We'll also import some helper functions that will be useful later on.
from util import load_data, cindex
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)


from sklearn.ensemble import GradientBoostingClassifier

viz_folder = "/content/train/plots"
model_path = "/content/models"
if os.path.exists(viz_folder):
  print("visulaization folder exist")
else:
  print("visulaization folder does not exist...making new folder")
  os.makedirs(viz_folder)

if os.path.exists(model_path):
  print("models folder exist")
else:
  print("models folder does not exist...making new folder")
  os.makedirs(model_path)


def fraction_rows_missing(df):
    '''
    Return percent of rows with any missing
    data in the dataframe. 
    
    Input:
        df (dataframe): a pandas dataframe with potentially missing data
    Output:
        frac_missing (float): fraction of rows with missing data
    '''
    ### START CODE HERE (REPLACE 'Pass' with your 'return' code) ###
    print("col:" ,sum(df.isnull().any(axis = 1)))
    print("row :" ,len(df.index))
    
    return sum(df.isnull().any(axis = 1))/ len(df.index)






X_dev, X_test, y_dev, y_test = load_data(10)

X_train, X_val, y_train, y_val = train_test_split(X_dev, 
                                                  y_dev, 
                                                  test_size=0.25,
                                                  random_state=10)

print("X_train shape: {}".format(X_train.shape))
print(X_train.head())

print("lables : /n", y_train.head())


i = 30
print(X_train.iloc[i,:])
print("\nDied within 10 years for the patient no:30? {}".format(y_train.loc[y_train.index[i]]))

sns.heatmap(X_train.isnull(), cbar=False)
plt.title("Training")
plt.savefig('/content/train/plots/MissingDataheatmap_training_data')
plt.close()

sns.heatmap(X_val.isnull(), cbar=False)
plt.title("Validation")
plt.savefig("/content/train/plots/MissingDataheatmap_val_data")
plt.close()


X_train_dropped = X_train.dropna(axis='rows')
y_train_dropped = y_train.loc[X_train_dropped.index]
X_val_dropped = X_val.dropna(axis='rows')
y_val_dropped = y_val.loc[X_val_dropped.index]



dt = DecisionTreeClassifier(max_depth=None, random_state=10)
dt.fit(X_train_dropped, y_train_dropped)

y_train_preds = dt.predict_proba(X_train_dropped)[:, 1]
print(f"DecisionTreeClassifier : Train C-Index: {cindex(y_train_dropped.values, y_train_preds)}")
y_val_preds = dt.predict_proba(X_val_dropped)[:, 1]
print(f"DecisionTreeClassifier : Val C-Index: {cindex(y_val_dropped.values, y_val_preds)}")

# Experiment with different hyperparameters for the DecisionTreeClassifier
# until you get a c-index above 0.6 for the validation set
dt_hyperparams = {
    # set your own hyperparameters below, such as 'min_samples_split': 1
    'max_depth': 3,
    'min_samples_split': 2,
}

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
dt_reg = DecisionTreeClassifier(**dt_hyperparams, random_state=10)
dt_reg.fit(X_train_dropped, y_train_dropped)

y_train_preds = dt_reg.predict_proba(X_train_dropped)[:, 1]
y_val_preds = dt_reg.predict_proba(X_val_dropped)[:, 1]
print(f" DecisionTreeClassifier : Train C-Index: {cindex(y_train_dropped.values, y_train_preds)}")
print(f" DecisionTreeClassifier : Val C-Index (expected > 0.6): {cindex(y_val_dropped.values, y_val_preds)}")

dot_data = StringIO()
export_graphviz(dt_reg, feature_names=X_train_dropped.columns, out_file=dot_data,  
                filled=True, rounded=True, proportion=True, special_characters=True,
                impurity=False, class_names=['neg', 'pos'], precision=2)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())




############################################### RANDOM FOREST ################################################### 

def holdout_grid_search(clf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparams, fixed_hyperparams={}):
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
        hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                            names to range of values for grid search
        fixed_hyperparams (dict): dictionary of fixed hyperparameters that
                                  are not included in the grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                             validation set
        best_hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                 names to values in best_estimator
    '''
    best_estimator = None
    best_hyperparams = {}
    
    # hold best running score
    best_score = 0.0

    # get list of param values
    lists = hyperparams.values()
    
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)

    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        # create estimator with specified params
        estimator = clf(**param_dict, **fixed_hyperparams)

        # fit estimator
        estimator.fit(X_train_hp, y_train_hp)
        
        # get predictions on validation set
        preds = estimator.predict_proba(X_val_hp)
        
        # compute cindex for predictions
        estimator_score = cindex(y_val_hp, preds[:,1])

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Val C-Index: {estimator_score}\n')

        # if new high score, update high score, best estimator
        # and best params 
        if estimator_score >= best_score:
                best_score = estimator_score
                best_estimator = estimator
                best_hyperparams = param_dict

    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_estimator, best_hyperparams


def random_forest_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped):

    # Define ranges for the chosen random forest hyperparameters 
    hyperparams = {
        
        ### START CODE HERE (REPLACE array values with your code) ###

        # how many trees should be in the forest (int)
        'n_estimators': [200, 300],

        # the maximum depth of trees in the forest (int)
        
        'max_depth': [90, 80],
        
        # the minimum number of samples in a leaf as a fraction
        # of the total number of samples in the training set
        # Can be int (in which case that is the minimum number)
        # or float (in which case the minimum is that fraction of the
        # number of training set samples)
        'min_samples_leaf': [1],

        ### END CODE HERE ###
    }

    
    fixed_hyperparams = {
        'random_state': 10,
    }
    
    rf = RandomForestClassifier

    best_rf, best_hyperparams = holdout_grid_search(rf, X_train_dropped, y_train_dropped,
                                                    X_val_dropped, y_val_dropped, hyperparams,
                                                    fixed_hyperparams)

    print(f"Best hyperparameters:\n{best_hyperparams}")

    
    y_train_best = best_rf.predict_proba(X_train_dropped)[:, 1]
    print(f"Train C-Index: {cindex(y_train_dropped, y_train_best)}")

    y_val_best = best_rf.predict_proba(X_val_dropped)[:, 1]
    print(f"Val C-Index: {cindex(y_val_dropped, y_val_best)}")
    
    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_rf, best_hyperparams


rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X_train_dropped, y_train_dropped)

y_train_rf_preds = rf.predict_proba(X_train_dropped)[:, 1]
print(f"RF : Train C-Index: {cindex(y_train_dropped.values, y_train_rf_preds)}")

y_val_rf_preds = rf.predict_proba(X_val_dropped)[:, 1]
print(f"RF : Val C-Index: {cindex(y_val_dropped.values, y_val_rf_preds)}")


best_rf, best_hyperparams = random_forest_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped)
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
y_test_best = best_rf.predict_proba(X_test)[:, 1]

print(f"RF : Test C-Index: {cindex(y_test.values, y_test_best)}")





############################################ GRADIENT BOOSTING ################################################################# 



def gradient_boosting_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped):

    # Define ranges for the chosen random forest hyperparameters 
    hyperparams = {
        
        ### START CODE HERE (REPLACE array values with your code) ###

        # how many trees should be in the forest (int)
        'n_estimators': [200, 300,400, 600],

        # the maximum depth of trees in the forest (int)
        
        'max_depth': [5,10,20,50],

        'learning_rate':[0.1 , 1.0 , 0.2],
        
        # the minimum number of samples in a leaf as a fraction
        # of the total number of samples in the training set
        # Can be int (in which case that is the minimum number)
        # or float (in which case the minimum is that fraction of the
        # number of training set samples)
        'min_samples_leaf': [1],

        ### END CODE HERE ###
    }

    
    fixed_hyperparams = {
        'random_state': 10,
    }
    
    gb = GradientBoostingClassifier

    best_gb, best_hyperparams = holdout_grid_search(gb, X_train_dropped, y_train_dropped,
                                                    X_val_dropped, y_val_dropped, hyperparams,
                                                    fixed_hyperparams)

    print(f"Best hyperparameters:\n{best_hyperparams}")

    
    y_train_best = best_gb.predict_proba(X_train_dropped)[:, 1]
    print(f"Train C-Index: {cindex(y_train_dropped, y_train_best)}")

    y_val_best = best_gb.predict_proba(X_val_dropped)[:, 1]
    print(f"Val C-Index: {cindex(y_val_dropped, y_val_best)}")
    
    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_gb, best_hyperparams


gb = GradientBoostingClassifier(learning_rate = 0.1 , n_estimators=400, random_state=10)
gb.fit(X_train_dropped, y_train_dropped)

y_train_gb_preds = gb.predict_proba(X_train_dropped)[:, 1]
print(f"GradientBoosting  : Train C-Index: {cindex(y_train_dropped.values, y_train_rf_preds)}")

y_val_gb_preds = gb.predict_proba(X_val_dropped)[:, 1]
print(f"GradientBoosting  : Val C-Index: {cindex(y_val_dropped.values, y_val_rf_preds)}")


best_gb, best_hyperparams = gradient_boosting_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped)
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)

y_test_best = best_gb.predict_proba(X_test)[:, 1]

print(f"GradientBoosting  : Test C-Index: {cindex(y_test.values, y_test_best)}")




###################################### IMPUTATION ###########################################################



# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def bad_subset(forest, X_test, y_test):
    # define mask to select large subset with poor performance
    # currently mask defines the entire set
    
    ### START CODE HERE (REPLACE the code after 'mask =' with your code) ###
    mask =X_test['Race'] < 30

    X_subgroup = X_test[mask]
    y_subgroup = y_test[mask]
    subgroup_size = len(X_subgroup)

    y_subgroup_preds = forest.predict_proba(X_subgroup)[:, 1]
    performance = cindex(y_subgroup.values, y_subgroup_preds)
    
    return performance, subgroup_size

dropped_rows = X_train[X_train.isnull().any(axis=1)]

columns_except_Systolic_BP = [col for col in X_train.columns if col not in ['Systolic BP']]

for col in columns_except_Systolic_BP:
    sns.distplot(X_train.loc[:, col], norm_hist=True, kde=False, label='full data')
    sns.distplot(dropped_rows.loc[:, col], norm_hist=True, kde=False, label='without missing data')
    plt.legend()
    plt.savefig('/content/train/plots/imputation' + str(col))
    #plt.show()
    plt.close()

# for random forest #
performance, subgroup_size = bad_subset(best_rf, X_test, y_test)
print("RF :Subgroup size should greater than 250, performance should be less than 0.69 ")
print(f"RF : Subgroup size: {subgroup_size}, C-Index: {performance}")

# for gradient boosting #
performance, subgroup_size = bad_subset(best_gb, X_test, y_test)
print("GB : Subgroup size should greater than 250, performance should be less than 0.69 ")
print(f"GB : Subgroup size: {subgroup_size}, C-Index: {performance}")



# Impute values using the mean
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)
X_train_mean_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_val_mean_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)






# Define ranges for the random forest hyperparameter search 
hyperparams = {
    ### START CODE HERE (REPLACE array values with your code) ###

    # how many trees should be in the forest (int)
    'n_estimators': [300],

    # the maximum depth of trees in the forest (int)
    'max_depth': [90],

    # the minimum number of samples in a leaf as a fraction
    # of the total number of samples in the training set
    # Can be int (in which case that is the minimum number)
    # or float (in which case the minimum is that fraction of the
    # number of training set samples)
    'min_samples_leaf': [3],

    ### END CODE HERE ###
}

# Define ranges for the chosen random forest hyperparameters 
hyperparams_gb = {'n_estimators': [400], 'max_depth': [5],
                  'learning_rate': [0.1], 'min_samples_leaf': [1]}


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
rf = RandomForestClassifier

rf_mean_imputed, best_hyperparams_mean_imputed = holdout_grid_search(rf, X_train_mean_imputed, y_train,
                                                                     X_val_mean_imputed, y_val,
                                                                     hyperparams, {'random_state': 10})

print("Performance for best hyperparameters:")

y_train_best = rf_mean_imputed.predict_proba(X_train_mean_imputed)[:, 1]
print(f"- Train C-Index: {cindex(y_train, y_train_best):.4f}")

y_val_best = rf_mean_imputed.predict_proba(X_val_mean_imputed)[:, 1]
print(f"- Val C-Index: {cindex(y_val, y_val_best):.4f}")

y_test_imp = rf_mean_imputed.predict_proba(X_test)[:, 1]
print(f"- Test C-Index: {cindex(y_test, y_test_imp):.4f}")




# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
gb = GradientBoostingClassifier

gb_mean_imputed, best_hyperparams_mean_imputed = holdout_grid_search(gb, X_train_mean_imputed, y_train,
                                                                     X_val_mean_imputed, y_val,
                                                                     hyperparams_gb, {'random_state': 10})

print("Performance for best hyperparameters gb:")

y_train_best = rf_mean_imputed.predict_proba(X_train_mean_imputed)[:, 1]
print(f"- gb Train C-Index: {cindex(y_train, y_train_best):.4f}")

y_val_best = rf_mean_imputed.predict_proba(X_val_mean_imputed)[:, 1]
print(f"- gb Val C-Index: {cindex(y_val, y_val_best):.4f}")

y_test_imp = rf_mean_imputed.predict_proba(X_test)[:, 1]
print(f"- gb Test C-Index: {cindex(y_test, y_test_imp):.4f}")








# Impute using regression on other covariates
imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=1, min_value=0)
imputer.fit(X_train)
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)





# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
rf = RandomForestClassifier

rf_imputed, best_hyperparams_imputed = holdout_grid_search(rf, X_train_imputed, y_train,
                                                           X_val_imputed, y_val,
                                                           hyperparams, {'random_state': 10})

print("Performance for best hyperparameters:")

y_train_best = rf_imputed.predict_proba(X_train_imputed)[:, 1]
print(f"- Train C-Index: {cindex(y_train, y_train_best):.4f}")

y_val_best = rf_imputed.predict_proba(X_val_imputed)[:, 1]
print(f"- Val C-Index: {cindex(y_val, y_val_best):.4f}")

y_test_imp = rf_imputed.predict_proba(X_test)[:, 1]
print(f"- Test C-Index: {cindex(y_test, y_test_imp):.4f}")



### gradient boosting iterative imputaion:
gb = GradientBoostingClassifier

gb_imputed, best_hyperparams_imputed = holdout_grid_search(gb, X_train_imputed, y_train,
                                                           X_val_imputed, y_val,
                                                           hyperparams_gb, {'random_state': 10})

print("Performance for best hyperparameters gb :")

y_train_best = gb_imputed.predict_proba(X_train_imputed)[:, 1]
print(f"- gb Train C-Index: {cindex(y_train, y_train_best):.4f}")

y_val_best = gb_imputed.predict_proba(X_val_imputed)[:, 1]
print(f"- gb Val C-Index: {cindex(y_val, y_val_best):.4f}")

y_test_imp = gb_imputed.predict_proba(X_test)[:, 1]
print(f"- gb Test C-Index: {cindex(y_test, y_test_imp):.4f}")





performance01, subgroup_size = bad_subset(best_rf, X_test, y_test)
print(f"C-Index rf (no imputation): {performance01}")

performance02, subgroup_size = bad_subset(rf_mean_imputed, X_test, y_test)
print(f"C-Index rf (mean imputation): {performance02}")

performance03, subgroup_size = bad_subset(rf_imputed, X_test, y_test)
print(f"C-Index rf (multivariate feature imputation): {performance03}")

performance04, subgroup_size = bad_subset(gb_mean_imputed, X_test, y_test)
print(f"C-Index rf (mean imputation): {performance04}")

performance05, subgroup_size = bad_subset(gb_imputed, X_test, y_test)
print(f"C-Index gb (multivariate feature imputation): {performance05}")

X = ['rf-no imputation' ,'rf-mean imputation' , 'rf-multivariate imputation',
      'gb-mean imputation' , 'gb-multivariate imputation']

Y = [performance01 , performance02, performance03, performance04, performance05]
plt.close()
plt.title('C-INDEX')
plt.bar(X,Y)
plt.savefig('/content/train/plots/comparison')



from joblib import dump, load
dump(rf_imputed, model_path + '/rf_imputed.joblib')