import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.stats
from sklearn.ensemble import   RandomForestClassifier
from sklearn.ensemble import   GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
import itertools
import activity_classifier_utils



fs = 256
data = activity_classifier_utils.LoadWristPPGDataset()
labels, subjects, features = activity_classifier_utils.GenerateFeatures(data,
                                                                        fs,
                                                                        window_length_s=10,
                                                                        window_shift_s=10)


n_estimators_opt = [2, 10, 20, 50, 100, 150, 300, 350, 400]
max_tree_depth_opt = range(2, 10)
learning_rate_opt  = [1, 0.1, 0.01]


class_names = np.array(['bike', 'run', 'walk'])
logo = LeaveOneGroupOut()
accuracy_table = []



#----------------------------------------Random Forest ---------------------------------------------------#
for n_estimators, max_tree_depth in itertools.product(n_estimators_opt, max_tree_depth_opt):
    # Iterate over each pair of hyperparameters
    cm = np.zeros((3, 3), dtype='int')                       # Create a new confusion matrix
    clf = RandomForestClassifier(n_estimators=n_estimators,  # and a new classifier  for each
                                 max_depth=max_tree_depth,   # pair of hyperparameters
                                 random_state=42,
                                 class_weight = "balanced"
                                )
    for train_ind, test_ind in logo.split(features, labels, subjects):
        # Do leave-one-subject-out cross validation as before.
        X_train, y_train = features[train_ind], labels[train_ind]
        X_test, y_test = features[test_ind], labels[test_ind]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c = confusion_matrix(y_test, y_pred, labels=class_names)
        cm += c
    # For each pair of hyperparameters, compute the classification accuracy
    classification_accuracy = np.sum(np.diag(cm)) / np.sum(np.sum(cm))
    
    # Store the hyperparameters and the classification accuracy that resulted
    # from the model created with them.
    accuracy_table.append(("random forest", n_estimators, max_tree_depth, classification_accuracy))


#------------------------------------Gradient Boosting trees-------------------------------------------#
print("training gradient bossting")

for n_estimators, max_tree_depth in itertools.product(n_estimators_opt, max_tree_depth_opt):
    # Iterate over each pair of hyperparameters
    cm = np.zeros((3, 3), dtype='int')                       # Create a new confusion matrix
    clf = GradientBoostingClassifier(n_estimators=n_estimators,  # and a new classifier  for each
                                 max_depth=max_tree_depth,   # pair of hyperparameters
                                 random_state=42,
                                )
    for train_ind, test_ind in logo.split(features, labels, subjects):
        # Do leave-one-subject-out cross validation as before.
        X_train, y_train = features[train_ind], labels[train_ind]
        X_test, y_test = features[test_ind], labels[test_ind]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c = confusion_matrix(y_test, y_pred, labels=class_names)
        cm += c
    # For each pair of hyperparameters, compute the classification accuracy
    classification_accuracy = np.sum(np.diag(cm)) / np.sum(np.sum(cm))
    
    # Store the hyperparameters and the classification accuracy that resulted
    # from the model created with them.
    accuracy_table.append(("Gradient Boosting",n_estimators, max_tree_depth, classification_accuracy))

# ----------------------------------------------------------AdaBoostClassifier-------------------------------------- #
print("training AdaBoostClassifier")


for n_estimators, learning_rate in itertools.product(n_estimators_opt, learning_rate_opt):
    # Iterate over each pair of hyperparameters
    cm = np.zeros((3, 3), dtype='int')                       # Create a new confusion matrix
    clf = AdaBoostClassifier(n_estimators=n_estimators,  # and a new classifier  for each
                              learning_rate = learning_rate,   # pair of hyperparameters
                                 random_state=42
                                )
    for train_ind, test_ind in logo.split(features, labels, subjects):
        # Do leave-one-subject-out cross validation as before.
        X_train, y_train = features[train_ind], labels[train_ind]
        X_test, y_test = features[test_ind], labels[test_ind]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c = confusion_matrix(y_test, y_pred, labels=class_names)
        cm += c
    # For each pair of hyperparameters, compute the classification accuracy
    classification_accuracy = np.sum(np.diag(cm)) / np.sum(np.sum(cm))
    
    # Store the hyperparameters and the classification accuracy that resulted
    # from the model created with them.
    accuracy_table.append(("AdaBoostClassifier", n_estimators, learning_rate, classification_accuracy))




accuracy_table_df = pd.DataFrame(accuracy_table,
                                 columns=["clf", 'n_estimators', 'max_tree_depth', 'accuracy'])  
accuracy_table_df.to_csv("file_name.csv" , index=False)  


# ------------------------------------------DecisionTreeClassifier-------------------------------------------#
print("training DecisionTreeClassifier")

for n_estimators, max_tree_depth in itertools.product(n_estimators_opt, max_tree_depth_opt):
    # Iterate over each pair of hyperparameters
    cm = np.zeros((3, 3), dtype='int')                       # Create a new confusion matrix
    clf = DecisionTreeClassifier(max_depth=max_tree_depth,   # pair of hyperparameters
                                 random_state=42,
                                )
    for train_ind, test_ind in logo.split(features, labels, subjects):
        # Do leave-one-subject-out cross validation as before.
        X_train, y_train = features[train_ind], labels[train_ind]
        X_test, y_test = features[test_ind], labels[test_ind]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c = confusion_matrix(y_test, y_pred, labels=class_names)
        cm += c
    # For each pair of hyperparameters, compute the classification accuracy
    classification_accuracy = np.sum(np.diag(cm)) / np.sum(np.sum(cm))
    
    # Store the hyperparameters and the classification accuracy that resulted
    # from the model created with them.
    accuracy_table.append(("DecisionTreeClassifier","None", max_tree_depth, classification_accuracy))









accuracy_table_df = pd.DataFrame(accuracy_table,
                                 columns=["clf", 'n_estimators', 'max_tree_depth', 'accuracy'])
print(accuracy_table_df.head(10))


accuracy_table_df.to_csv("file_name.csv" , index=False) 
accuracy_table_df.loc[accuracy_table_df.accuracy.idxmax()]
print("best hyperparameters : ", accuracy_table_df.loc[accuracy_table_df.accuracy.idxmax()])









#----------------------------------------NESTED CROSS VALIDATIONS ----------------------------------------------#
class_names = ['bike', 'run', 'walk']

# Store the confusion matrix for the outer CV fold.
nested_cv_cm = np.zeros((3, 3), dtype='int')
splits = 0

for train_val_ind, test_ind in logo.split(features, labels, subjects):
    # Split the dataset into a test set and a training + validation set.
    # Model parameters (the random forest tree nodes) will be trained on the training set.
    # Hyperparameters (how many trees and the max depth) will be trained on the validation set.
    # Generalization error will be computed on the test set.
    X_train_val, y_train_val = features[train_val_ind], labels[train_val_ind]
    subjects_train_val = subjects[train_val_ind]
    X_test, y_test = features[test_ind], labels[test_ind]
    
    # Keep track of the best hyperparameters for this training + validation set.
    best_hyper_parames = None
    best_accuracy = 0
    
    for n_estimators, max_tree_depth in itertools.product(n_estimators_opt,
                                                          max_tree_depth_opt):
        # Optimize hyperparameters as above.
        inner_cm = np.zeros((3, 3), dtype='int')
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_tree_depth,
                                     random_state=42,
                                     class_weight='balanced')
        for train_ind, validation_ind in logo.split(X_train_val, y_train_val,
                                                    subjects_train_val):
            X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
            X_val, y_val = X_train_val[validation_ind], y_train_val[validation_ind]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            c = confusion_matrix(y_val, y_pred, labels=class_names)
            inner_cm += c
        classification_accuracy = np.sum(np.diag(inner_cm)) / np.sum(np.sum((inner_cm)))
        
        # Keep track of the best pair of hyperparameters.
        if classification_accuracy > best_accuracy:
            best_accuracy = classification_accuracy
            best_hyper_params = (n_estimators, max_tree_depth)
    
    # Create a model with the best pair of hyperparameters for this training + validation set.
    best_clf = RandomForestClassifier(n_estimators=best_hyper_params[0],
                                      max_depth=best_hyper_params[1],
                                      class_weight='balanced')
    
    # Finally, train this model and test it on the test set.
    best_clf.fit(X_train_val, y_train_val)
    y_pred = best_clf.predict(X_test)
    
    # Aggregate confusion matrices for each CV fold.
    c = confusion_matrix(y_test, y_pred, labels=class_names)
    nested_cv_cm += c
    splits += 1
    print('Done split {}'.format(splits))

acc = np.sum(np.diag(nested_cv_cm)) / np.sum(np.sum(nested_cv_cm))
print("classification accuracy" , acc)



clf = RandomForestClassifier(n_estimators=100,
                             max_depth=4,
                             random_state=42,
                             class_weight='balanced')
activity_classifier_utils.LOSOCVPerformance(features, labels, subjects, clf)
feat = clf.feature_importances_
print(len(feat))

plt.figure(figsize=(15,8))
plt.xticks(rotation=45)
plt.bar(x =activity_classifier_utils.FeatureNames(), height= feat)

plt.savefig("feature_importance")

print( sorted(list(zip(clf.feature_importances_, 
activity_classifier_utils.FeatureNames())), reverse=True)[:10]
)

sorted_features = sorted(zip(clf.feature_importances_, np.arange(len(clf.feature_importances_))), reverse=True)
best_feature_indices = list(zip(*sorted_features))[1]
X = features[:, best_feature_indices[:10]]

print("Shape of the input data :" , X.shape)

cm = activity_classifier_utils.LOSOCVPerformance(X, labels, subjects, clf)
plt.close()
activity_classifier_utils.PlotConfusionMatrix(cm, class_names, normalize=True)
plt.savefig("confusion_matrix_normalized")
print('Classification accuracy = {:0.2f}'.format(np.sum(np.diag(cm)) / np.sum(np.sum(cm))))


