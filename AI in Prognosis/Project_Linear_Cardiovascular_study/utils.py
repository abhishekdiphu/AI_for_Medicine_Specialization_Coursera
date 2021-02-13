import numpy as np
import pandas as pd
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## preprocessing pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns








def make_standard_normal(df_train, df_test , features, categorical):
    """
    In order to make the data closer to a normal distribution, take log
    transforms to reduce the skew.
    Then standardize the distribution with a mean of zero and standard deviation of 1. 
  
    Args:
      df_train (dataframe): unnormalized training data.
      df_test (dataframe): unnormalized test data.
  
    Returns:
      df_train_normalized (dateframe): normalized training data.
      df_test_normalized (dataframe): normalized test data.
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###  
    # Remove skew by applying the log function to the train set, and to the test set
    df_train_unskewed = np.log(df_train[features])
    df_test_unskewed =  np.log(df_test[features])
    
    #calculate the mean and standard deviation of the training set
    mean = df_train_unskewed.mean(axis = 0)
    stdev =df_train_unskewed.std(axis = 0)
    categorical = categorical
    # standardize the training set
    df_train_standardized = (df_train_unskewed - mean)/ stdev

    df_train_standardized =pd.concat([df_train[categorical] ,df_train_standardized], axis=1)
    # standardize the test set (see instructions and hints above)
    df_test_standardized = (df_test_unskewed - mean)/ stdev
    df_test_standardized = pd.concat([df_test[categorical] ,df_test_standardized], axis=1)
    
    ### END CODE HERE ###
    return df_train_standardized, df_test_standardized
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)




def add_interactions(X):
    """
    Add interaction terms between columns to dataframe.

    Args:
    X (dataframe): Original data

    Returns:
    X_int (dataframe): Original data with interaction terms appended. 
    """
    features = X.columns
    m = len(features)
    X_int = X.copy(deep=True)


    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # 'i' loops through all features in the original dataframe X
    for i in range(m):
        
        # get the name of feature 'i'
        feature_i_name = features[i]
        
        # get the data for feature 'i'
        feature_i_data =X_int[feature_i_name]
        
        # choose the index of column 'j' to be greater than column i
        for j in range(i+1, m):
            
            # get the name of feature 'j'
            feature_j_name = features[j]
            
            # get the data for feature j'
            feature_j_data = X_int[feature_j_name]
            
            # create the name of the interaction feature by combining both names
            # example: "apple" and "orange" are combined to be "apple_x_orange"
            feature_i_j_name =  f"{feature_i_name}_x_{feature_j_name}"
            print(feature_i_j_name)
            
            # Multiply the data for feature 'i' and feature 'j'
            # store the result as a column in dataframe X_int
            X_int[feature_i_j_name] = (feature_i_data) * (feature_j_data)
        
    ### END CODE HERE ###

    return X_int
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def cindex(y_true, scores):
    '''

    Input:
    y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
        0: patient does not get the disease
        1: patient does get the disease
    scores (np.array): a 1-D array of corresponding risk scores output by the model

    Output:
    c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
    '''
    n = len(y_true)
    assert len(scores) == n

    concordant = 0
    permissible = 0
    ties = 0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # use two nested for loops to go through all unique pairs of patients
    for i in range(n):
        for j in range(i+1, n): #choose the range of j so that j>i
            
            # Check if the pair is permissible (the patient outcomes are different)
            if y_true[i] != y_true[j]:
                # Count the pair if it's permissible
                permissible+=1

                # For permissible pairs, check if they are concordant or are ties

                # check for ties in the score
                if y_true[i] != y_true[j] and scores[i] == scores[j]:
                    # count the tie
                    ties +=1
                    # if it's a tie, we don't need to check patient outcomes, continue to the top of the for loop.
                    continue

                # case 1: patient i doesn't get the disease, patient j does
                if y_true[i] == 0 and y_true[j] == 1:
                    # Check if patient i has a lower risk score than patient j
                    if scores[i] < scores[j]:
                        # count the concordant pair
                        concordant +=1
                    # Otherwise if patient i has a higher risk score, it's not a concordant pair.
                    # Already checked for ties earlier

                # case 2: patient i gets the disease, patient j does not
                if y_true[i] == 1 and y_true[j] == 0:
                    # Check if patient i has a higher risk score than patient j
                    if scores[i] > scores[j]:
                        #count the concordant pair
                        concordant +=1
                    # Otherwise if patient i has a lower risk score, it's not a concordant pair.
                    # We already checked for ties earlier

    # calculate the c-index using the count of permissible pairs, concordant pairs, and tied pairs.
    c_index = (concordant + 0.5*ties) / permissible 
    ### END CODE HERE ###
    
    return c_index


def imputer(numeric_features , categorical_features):
        numeric_features = numeric_features
        numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='mean')),
                                      ('scaler', StandardScaler())])

        categorical_features = categorical_features
        categorical_transformer =  Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('encod', OneHotEncoder())])

        preprocessor = ColumnTransformer(transformers=[
                                        ('num', numeric_transformer, numeric_features),
                                        ('cat', categorical_transformer, categorical_features)])
        
        #transformer = preprocessor.fit(X)
        #X = transformer.transform(X)
        return preprocessor



def generate_data(n=200):
    df = pd.DataFrame(
        columns=['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']
    )
    df.loc[:, 'Age'] = np.exp(np.log(60) + (1 / 7) * np.random.normal(size=n))
    df.loc[:, ['Systolic_BP', 'Diastolic_BP', 'Cholesterol']] = np.exp(
        np.random.multivariate_normal(
            mean=[np.log(100), np.log(90), np.log(100)],
            cov=(1 / 45) * np.array([
                [0.5, 0.2, 0.2],
                [0.2, 0.5, 0.2],
                [0.2, 0.2, 0.5]]),
            size=n))
    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f(x):
    p = 0.4 * (np.log(x[0]) - np.log(60)) + 0.33 * (
            np.log(x[1]) - np.log(100)) + 0.3 * (
                np.log(x[3]) - np.log(100)) - 2.0 * (
                np.log(x[0]) - np.log(60)) * (
                np.log(x[3]) - np.log(100)) + 0.05 * np.random.logistic()
    if p > 0.0:
        return 1.0
    else:
        return 0.0


def load_data(n=200):
    np.random.seed(0)
    df = generate_data(n)
    for i in range(len(df)):
        df.loc[i, 'y'] = f(df.loc[i, :])
        X = df.drop('y', axis=1)
        y = df.y
    return X, y
