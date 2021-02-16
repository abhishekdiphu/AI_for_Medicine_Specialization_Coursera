import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *
from utils import load_data
from helper import *

# This function creates randomly generated data
# X, y = load_data(6000)

# For stability, load data from files that were generated using the load_data
X = pd.read_csv('X_data.csv',index_col=0)
y_df = pd.read_csv('y_data.csv',index_col=0)
y = y_df['y']


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


for col in X.columns:
    X_train_raw.loc[:, col].hist()
    plt.title(col)
    plt.show()
    plt.savefig(col)
    plt.close()
    plt.savefig(col + 'fig')
    plt.close()
    
X_train, X_test = make_standard_normal(X_train_raw, X_test_raw)

model_X = lr_model(X_train, y_train)


scores = model_X.predict_proba(X_test)[:, 1]
c_index_X_test = cindex(y_test.values, scores)

print(f"c-index on test set is {c_index_X_test:.4f}")


coeffs = pd.DataFrame(data = model_X.coef_, columns = X_train.columns)
coeffs.T.plot.bar(legend=True)
plt.savefig("coef.png")


scores_X = model_X.predict_proba(X_test)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X)







X_train_int = add_interactions(X_train)
X_test_int = add_interactions(X_test)
model_X_int = lr_model(X_train_int, y_train)

scores_X = model_X.predict_proba(X_test)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X)

scores_X_int = model_X_int.predict_proba(X_test_int)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X_int)

print(f"c-index on test set without interactions is {c_index_X_test:.4f}")
print(f"c-index on test set with interactions is {c_index_X_int_test:.4f}")


plt.close()
int_coeffs = pd.DataFrame(data = model_X_int.coef_, columns = X_train_int.columns)
int_coeffs.T.plot.bar();
plt.savefig('interection_term.png')


index = index = 3432
case = X_train_int.iloc[index, :]
print(case)


new_case = case.copy(deep=True)
new_case.loc["Age_x_Cholesterol"] = 0
new_case


print(f"Output with interaction: \t{model_X_int.predict_proba([case.values])[:, 1][0]:.4f}")
print(f"Output without interaction: \t{model_X_int.predict_proba([new_case.values])[:, 1][0]:.4f}")

    

    