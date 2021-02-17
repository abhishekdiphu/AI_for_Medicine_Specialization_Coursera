#!R -e 'install.packages(c("randomForestSRC"), repos="https://cran.r-project.org", dependencies=TRUE)'


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as cindex
from sklearn.model_selection import train_test_split
from util import load_data
from model_utils import *



df = load_data()


print(df.shape)
print(df.head(2))

np.random.seed(0)
df_dev, df_test = train_test_split(df, test_size = 0.2)
df_train, df_val = train_test_split(df_dev, test_size = 0.25)

print("Total number of patients:", df.shape[0])
print("Total number of patients in training set:", df_train.shape[0])
print("Total number of patients in validation set:", df_val.shape[0])
print("Total number of patients in test set:", df_test.shape[0])


continuous_columns = ['age', 'bili', 'chol', 'albumin', 
                      'copper', 'alk.phos', 'ast', 'trig',
                      'platelet', 'protime']
mean = df_train.loc[:, continuous_columns].mean()
std = df_train.loc[:, continuous_columns].std()
df_train.loc[:, continuous_columns] = (df_train.loc[:, continuous_columns] - mean) / std
df_val.loc[:, continuous_columns] = (df_val.loc[:, continuous_columns] - mean) / std
df_test.loc[:, continuous_columns] = (df_test.loc[:, continuous_columns] - mean) / std


# List of categorical columns
to_encode = ['edema', 'stage']

one_hot_train = to_one_hot(df_train, to_encode)
one_hot_val = to_one_hot(df_val, to_encode)
one_hot_test = to_one_hot(df_test, to_encode)

print(one_hot_val.columns.tolist())
print(f"There are {len(one_hot_val.columns)} columns")



#%load_ext rpy2.ipython
#%R require(ggplot2)

from rpy2.robjects.packages import importr
# import R's "base" package

base = importr('base')

# import R's "utils" package
utils = importr('utils')

# import rpy2's package module
import rpy2.robjects.packages as rpackages

forest = rpackages.importr('randomForestSRC')

from rpy2 import robjects as ro
R = ro.r

from rpy2.robjects import pandas2ri
pandas2ri.activate()

model = forest.rfsrc(ro.Formula('Surv(time, status) ~ .'), 
          data=df_train,
          ntree= 800,
          nodedepth= 25,
          seed=-1)

print(model)

result = R.predict(model, newdata=df_val)
scores = np.array(result.rx('predicted')[0])

#print("Cox Model Validation Score:", cox_val_scores)

print("Survival Forest Validation Score:", 
        harrell_c(df_val['time'].values, 
        scores, df_val['status'].values))


result = R.predict(model, newdata=df_test)
scores = np.array(result.rx('predicted')[0])

# print("Cox Model Test Score:", cox_test_scores)

print("Survival Forest Validation Score:", 
        harrell_c(df_test['time'].values,
        scores, df_test['status'].values))


vimps = np.array(forest.vimp(model).rx('importance')[0])

y = np.arange(len(vimps))


fig = plt.figure(figsize =(10,10))
plt.barh(y, np.abs(vimps))
plt.yticks(y, df_train.drop(['time', 'status'], axis=1).columns)
plt.title("VIMP (absolute value)")
plt.show()
plt.savefig('premutation_feature_importance')
plt.close()