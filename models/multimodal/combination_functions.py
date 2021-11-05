import numpy as np
import scipy.special



'''
Defines possible combination functions which can be used by the LateFusionModel class
to combine predictions between models trained on different modalities.

All functions should have the same interface:
Parameters:
    predictions (np.ndarray with shape number_of_models, prediction_shape)
Returns:
    (float): Performance metric value
'''


'''
Get the mean of the predictions for model used for classification.
Assumes the component models all output probability distributions over the classes.
'''
def MEAN_CLASSIFICATION(predictions: np.ndarray) -> np.ndarray:
    # note: no sofmax needed b/c the mean over the values output for the distribution produces
    # another distribution. (you can prove this pretty easily)
    return np.mean(predictions, axis=0)
