# helper functions#

import numpy as np
from keras import backend as K

def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    df1_patients_unique =set(df1[patient_col].values) 
    df2_patients_unique =set(df2[patient_col].values) 
    
    patients_in_both_groups =df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    n_overlap =len(patients_in_both_groups) 
    leakage = n_overlap > 0  # boolean (true if there is at least 1 patient in both groups)
    
    ### END CODE HERE ###
    
    return leakage


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = None
    
    positive_frequencies =np.sum(labels == 1,axis=0) / labels.shape[0]
    negative_frequencies =np.sum(labels == 0,axis=0) / labels.shape[0]
    print(positive_frequencies)
    print(negative_frequencies)

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies





# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += K.mean(-1*(y_true[: , i])*K.log(y_pred[: , i] + epsilon)  + \
                     (-1)*(1 - y_true[: , i])*K.log(1 - y_pred[: , i] + epsilon)) #complete this line
        return loss
    
        ### END CODE HERE ###
    return weighted_loss