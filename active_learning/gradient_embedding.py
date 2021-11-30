from typing import List
import numpy as np
from collections import deque
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
Utility for creating gradient embeddings as detailed in BADGE and Cluster-Margin algorithms.
'''

'''
Given a model and parameter references, creates embeddings of each given point based on the gradients
of predicted cross-entropy loss with respect to the target parameters (parameters of last network layer).
Assumes model output is categorical (returns normalized softmax probabilities).
Overview:
    -   Assume actual label is same as max-probability predicted label for each sample, and compute
        cross-entropy loss against these assumed labels.
    -   Perform backpropagation on this loss.
    -   Embedding value for a sample = parameter gradients from assumed loss for that sample.
Args:
    model (nn.Module):                  Model for which to embed values. Must have 1D normalized categorical probabilities 
                                        as output (per sample).
    target_parameters ([nn.Parameter]): List of parameters to compute gradients for. Generally should be parameters of
                                        last feature layer.
    model_inputs ([np.ndarray]):        List of batch of samples in each mode which will be transformed into embeddings. Must be a valid input
                                        for model. List elements will be unpacked when input into model.
Returns:
    (np.ndarray):                       Embedding for each sample. Shape = (batch_size, flattened_param_size)
    (np.ndarray):                       List mapping embedding indices to their corresponding parameter indices
'''
def compute_gradient_embeddings(model: nn.Module, target_parameters: List[nn.Parameter], model_inputs: List[np.ndarray]) -> np.ndarray:
    target_parameters = list(target_parameters) # Ensure target_parameters is in list format, not Generator format
    
    embedding_reversal_map = [
        [i, *np.unravel_index(j, target_parameters[i].shape)]
        for i in range(len(target_parameters))
        for j in range(np.prod(target_parameters[i].shape))
    ]
    embedding_len = sum(np.prod(p.shape) for p in target_parameters)

    data_len = model_inputs[0].shape[0]

    embeddings = np.zeros((data_len, embedding_len))
    for i in range(data_len):
        # Convert individual sample to gpu tensor, as a batch of 1 sample
        input = [
            torch.from_numpy(input_mode[i:i+1]).to(DEVICE)
            for input_mode in model_inputs
        ]

        model.zero_grad()
        assumed_sample_loss = -torch.log(torch.max(model(*input)))
        sample_grads = torch.autograd.grad(assumed_sample_loss, target_parameters)
        flattened_sample_grads = torch.cat([g.reshape(-1) for g in sample_grads])
        embeddings[i, :] = flattened_sample_grads.detach().cpu().numpy()

    return embeddings, embedding_reversal_map


if __name__ == "__main__":
    # Quick test for formatting
    inputs = torch.randn(10, 2)
    model = nn.Linear(2, 3)
    params = list(model.parameters())
    embeddings = compute_gradient_embeddings(model, params, inputs)
    print(embeddings.shape)
    print(embeddings)

