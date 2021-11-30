from typing import List
import numpy as np
import torch
import torch.nn as nn

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
    samples (torch.FloatTensor):        Batch of samples which will be transformed into embeddings. Must be a valid input
                                        for model.
Returns:
    (torch.FloatTensor):                Embedding for each sample. Shape = (batch_size, flattened_param_size)
'''
def compute_gradient_embeddings(model: nn.Module, target_parameters: List[nn.Parameter], samples: torch.FloatTensor) -> torch.FloatTensor:
    embedding_len = 0
    for p in target_parameters:
        embedding_len += np.prod(p.shape)

    embeddings = torch.zeros(samples.shape[0], embedding_len)
    for i in range(samples.shape[0]):
        sample = samples[i]

        model.zero_grad()
        assumed_sample_loss = torch.log(torch.max(model(sample.reshape(1, *sample.shape))))
        sample_grads = torch.autograd.grad(assumed_sample_loss, target_parameters)
        flattened_sample_grads = torch.cat([g.reshape(-1) for g in sample_grads])
        embeddings[i, :] = flattened_sample_grads

    return embeddings


if __name__ == "__main__":
    # Quick test for formatting
    inputs = torch.randn(10, 2)
    model = nn.Linear(2, 3)
    params = list(model.parameters())
    embeddings = compute_gradient_embeddings(model, params, inputs)
    print(embeddings.shape)
    print(embeddings)

