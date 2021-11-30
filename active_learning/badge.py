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
        # Find current output probabilities
        outputs = self.model(torch.tensor(unlabeled_samples).to(DEVICE)).detach().cpu().numpy()

        # Select margin batch (most uncertain samples)
        margin_batch_indices = MIN_MARGIN(outputs, self.margin_batch_size)

        # Sort all chosen samples into their clusters
        clusters = {}
        for sample_ind in margin_batch_indices:
            sample = unlabeled_samples[sample_ind]

            if sample.tobytes() not in self.sample_to_cluster_id:
                raise ValueError("Given unlabeled input not in cluster member dict")

            cluster_id = self.sample_to_cluster_id[sample.tobytes()]

            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sample_ind)

        # Shuffle all clusters
        for cluster_id, sample_indices in clusters.items():
            random.shuffle(sample_indices)

        # Sort clusters by size
        sorted_cluster_indices = list(clusters.keys()).sort(key=lambda cluster_id: len(clusters[cluster_id]))

        # Fill labeling batch by iterating through clusters in ascending size order
        cluster_loop_counter = 0  # Counts the number of times we have iterated through all clusters
        label_batch = []
        while len(label_batch) < self.target_batch_size:
            for cluster_ind in sorted_cluster_indices:
                if cluster_loop_counter < len(clusters[cluster_ind]):
                    label_batch.append(clusters[cluster_ind][cluster_loop_counter])
                    if len(label_batch) >= self.target_batch_size:
                        break
            cluster_loop_counter += 1

        return np.array(label_batch)