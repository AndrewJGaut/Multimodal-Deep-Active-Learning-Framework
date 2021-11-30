"""
The BADGE active learning method.
"""
from typing import Tuple
import numpy as np
import sklearn
import random
import torch
import torch.nn as nn
from clustering.sampling import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


from .categorical_query_functions import *
from .gradient_embedding import compute_gradient_embeddings


class BADGEQueryFunction:
    '''
    Args:
        model (nn.Module):                          Pytorch model which converts batch of inputs
                                                    into batch of normalized class probabilities
        last_layer_model_params (nn.Parameter):     Reference to final layer parameters in model
        margin_batch_size (int):                    The number of points to select based on high
                                                    uncertainty
        target_batch_size (int):                    The number of points to select in one active
                                                    learning batch
    '''

    def __init__(self, model: nn.Module, last_layer_model_params: nn.Parameter, margin_batch_size: int,
                 target_batch_size: int, sample_method: SampleMethod) -> None:
        self.model = model
        self.last_layer_model_params = last_layer_model_params
        self.margin_batch_size = margin_batch_size
        self.target_batch_size = target_batch_size
        self.sample_method = sample_method


    '''
    Active Learning query function, returns a subset of the given unlabeled samples
    Args:
        unlabeled_samples (np.ndarray): List of unlabeled samples
    Returns:
        batch_to_label (np.ndarray):    Subset of given unlabeled samples chosen for labeling
    '''

    def query(self, unlabeled_samples: np.ndarray) -> np.ndarray:
        embeddings, _ = compute_gradient_embeddings(self.model, self.last_layer_model_params, unlabeled_samples)
        sample_indices = self.sample_method(embeddings, self.target_batch_size)

        return sample_indices