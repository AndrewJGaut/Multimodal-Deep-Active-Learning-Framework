import torch
import torch.nn as nn
import numpy as np

'''
Defines modular query functions (which don't require knowledge of model parameters)
which can be added on top of any base learning algorithm in a ModelInterface subclass.
These functions only apply to categorical probability outputs.
'''


'''
Queries the samples which have the lowest maximum probability.
Args:
    probabilities (np.ndarray): Base model output, shape = (n_unlabeled, n_categories)
    labeling_batch_size (int): Number of samples which should be selected for labeling
'''
def MIN_MAX(probabilities: np.ndarray, labeling_batch_size: int) -> np.ndarray:
    max_probabilities = np.max(probabilities, axis=1)
    indices = np.argsort(max_probabilities)[:labeling_batch_size]
    return indices


'''
Queries the samples which have the smallest margin between maximum and second-maximum probability.
Args:
    probabilities (np.ndarray): Base model output, shape = (n_unlabeled, n_categories)
    labeling_batch_size (int): Number of samples which should be selected for labeling
'''
def MIN_MARGIN(probabilities: np.ndarray, labeling_batch_size: int) -> np.ndarray:
    max_probabilities = np.max(probabilities, axis=1)
    probabilities_without_max = probabilities * (probabilities != max_probabilities.reshape(-1, 1))
    second_max_probabilities = np.max(probabilities_without_max, axis=1)
    margin = max_probabilities - second_max_probabilities
    indices = np.argsort(margin)[:labeling_batch_size]
    return indices

'''
Queries the samples which have the maximum probability entropy
Args:
    probabilities (np.ndarray): Base model output, shape = (n_unlabeled, n_categories)
    labeling_batch_size (int): Number of samples which should be selected for labeling
'''
def MAX_ENTROPY(probabilities: np.ndarray, labeling_batch_size: int) -> np.ndarray:
    eps = 1e-6 # To avoid divide by zero error in log
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    indices = np.argsort(entropy)[-labeling_batch_size:][::-1]
    return indices