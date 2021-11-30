from typing import Tuple
import numpy as np
import sklearn
import random
import torch
import torch.nn as nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


from .categorical_query_functions import *
from .gradient_embedding import compute_gradient_embeddings

'''
Cluster-Margin works in two stages:
- Embedding/Clustering: Given a partially trained model-state, embed all unlabeled points:
    - Assume label matches predicted label, and then the encoded point is just the gradient
    of the cross-entropy loss (with this imagined label) with respect the parameters of the penultimate layer
    - Cluster these embeddings using HAC
- Batch Selection:
    - Select all unlabeled samples with uncertainty (margin between max and 2nd-max outputs) below some
    threshold
    - Map those unlabeled samples into their respective embedding clusters (creating a set of clusters of
    high-uncertainty points)
    - Add points to batch by iterating through selected clusters in order of ascending cluster size, sampling
    one point from each cluster. Loop through clusters in this order (without replacing already-sampled points)
    until desired batch size is reached.
'''

class ClusterMarginQueryFunction:
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
    def __init__(self, model: nn.Module, last_layer_model_params: nn.Parameter, margin_batch_size: int, target_batch_size: int) -> None:
        self.model = model
        self.last_layer_model_params = last_layer_model_params
        self.margin_batch_size = margin_batch_size
        self.target_batch_size = target_batch_size

        # For storing cluster membership {np array byte string -> int}
        self.sample_to_cluster_id = {}

    '''
    Update stored clusters which will be used to enforce diversity
    Args:
        samples (np.ndarray):   All currently unlabeled samples.
    '''
    def compute_clusters(self, samples: np.ndarray) -> None:
        # Find embeddings
        embeddings = compute_gradient_embeddings(self.model, self.last_layer_model_params, samples)
        
        # Find clusters
        n_clusters = len(samples) * (self.target_batch_size / self.margin_batch_size)
        cluster_ids = cluster_embeddings(embeddings, n_clusters)
        
        # Save cluster assignments
        self.sample_to_cluster_id = {}
        for i in range(len(samples)):
            self.sample_to_cluster_id[samples[i].tobytes()] = cluster_ids[i]
    
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
        sorted_cluster_indices = list(clusters.keys()).sort(key = lambda cluster_id : len(clusters[cluster_id]))

        # Fill labeling batch by iterating through clusters in ascending size order
        cluster_loop_counter = 0 # Counts the number of times we have iterated through all clusters
        label_batch = []
        while len(label_batch) < self.target_batch_size:
            for cluster_ind in sorted_cluster_indices:
                if cluster_loop_counter < len(clusters[cluster_ind]):
                    label_batch.append(clusters[cluster_ind][cluster_loop_counter])
                    if len(label_batch) >= self.target_batch_size:
                        break
            cluster_loop_counter += 1
        
        return np.array(label_batch)


# --- Helper Functions ---

'''
Turns a set of points into embeddings based on the gradient of imagined loss with respect to
parameters of the penultimate layer in the model.
Args:
    model (nn.Module):                          Pytorch model which converts batch of inputs
                                                into batch of normalized class probabilities
    last_layer_model_params (nn.Parameter):     Reference to final layer parameters in model
    samples (np.ndarray):                       Set of samples to create embeddings for
Returns:
    embeddings (np.ndarray):                    Embeddings for each element of points
'''
def embed_points(model: nn.Module, last_layer_model_params: nn.Parameter, samples: np.ndarray) -> np.ndarray:
    raise NotImplementedError

'''
Performs Hierarchical Agglomerative Clustering with Average-Linking on the given embeddings,
returning the cluster id corresponding to each point, as well as the total number of clusters.
Args:
    embeddings (np.ndarray):            Set of embeddings corresponding to points which
                                        should be clustered
    n_clusters (int):                   Number of clusters to create
Returns:
    cluster_membership (np.ndarray):    Integer cluster indices for each element of embeddings
'''
def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, int]:
    return sklearn.cluster.AgglomerativeClustering(n_clusters = n_clusters, linkage = 'average').fit(embeddings).labels_

'''

'''