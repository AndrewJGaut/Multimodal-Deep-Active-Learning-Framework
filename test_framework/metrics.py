import numpy as np
import sklearn.metrics as skm


'''
Defines possible objective functions which can be used by the Tester class
to evaluate base model performance at various stages of training and active learning.

Needed for if the output format changes between datatypes

These functions can just be wrappers for functions from ML frameworks, etc, this just
makes it easy to find and add to the list.


All functions should have the same interface:
Args:
    y_actual (np.ndarray)
    y_predicted (np.ndarray)
Returns:
    (float): Performance metric value
'''


'''
Categorical prediction accuracy.
Assumes y_actual entries are 1-hot arrays, and y_predicted entries are
normalized probability arrays of the same length.
'''
def ACCURACY(y_actual:np.ndarray, y_predicted:np.ndarray) -> float:
    y_prediction_probs = np.max(y_predicted, axis=-1, keepdims=True)
    return np.mean(y_actual[y_predicted == y_prediction_probs])

'''
Categorical log likelihood (negative binary cross-entropy).
Assumes y_actual entries are 1-hot arrays, and y_predicted entries are
normalized probability arrays of the same length.
'''
def LOG_LIKELIHOOD(y_actual:np.ndarray, y_predicted:np.ndarray) -> float:
    return -skm.log_loss(y_actual, y_predicted)