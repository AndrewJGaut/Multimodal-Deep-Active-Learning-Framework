import torch


class SampleMethod:
    def __init__(self, method_name, n_samples):
        self.method_name = method_name
        self.n_samples = n_samples

    def sample(self, X, n_samples=None):
        """
        :param X (torch.tensor): the points to sample from
        :param n_samples (int): the number of points to sample
        :return: The sampled points. These should be selected for by balancing both diversity and magnitude
        """
        pass


class KMeansPlusPlusSeeding(SampleMethod):
    def __init__(self, n_samples):
        super("KMeansPlusPlusSeeding", n_samples).__init__()


    def sample(self, X, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        if n_samples == 0:
            return None

        # for the first sample, just get a random point in X
        # keep in mind that X is of shape (n_examples, n_dimension_in_grad_embedding)
        # so the cluster points are the ROWS of X.
        centers = X[torch.randint(low=0, high=X.shape[0], size=(1,))].reshape(1, 1, -1)

        while centers.shape[0] < n_samples:
            squared_distances = torch.min(torch.norm(X - centers, dim=2), dim=0).values ** 2
            multinomial_parameters = torch.nn.functional.softmax(squared_distances, -1)
            multinomial_parameters[squared_distances == 0] = 0 # make sure we don't re-select a previously selected center
            new_center_index = torch.multinomial(multinomial_parameters, 1)
            centers = torch.cat((centers, X[new_center_index].reshape(1, 1, -1)))

        # the samples are the computed centers.
        return centers

class WeightedKMeansSampling(SampleMethod):
    """
    What this does is it performs weighted k-means clustering (which basically up-weights points according to some
    criteria, thereby making them more likely to be cluster centers) and uses the cluster centers as the points
    sampled. The idea is that, if the weights represent the informativeness of a point, then the weighted cluster
    centers balance both diversity AND informativeness.
     For the gradient embeddings, the "inforamtiveness" measurement is literally just the L2 norm of the embedding
     for this implementation.
     See Diverse mini-batch Active Learning (https://arxiv.org/pdf/1901.05954.pdf)
    """

    def __init__(self, n_samples):
        super("WeightedKMeansSampling", n_samples).__init__()


    def sample(self, X, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        if n_samples == 0:
            return None

        # for the first sample, just get a random point in X
        # keep in mind that X is of shape (n_examples, n_dimension_in_grad_embedding)
        # so the cluster points are the ROWS of X.
        centers = X[torch.randint(low=0, high=X.shape[0], size=(1,))]

        while centers.shape[0] < n_samples:
            new_center_index = torch.multinomial(torch.nn.functional.softmax(centers[centers != 0]), 1)
            centers = torch.cat((centers, X[new_center_index].reshape(1,-1)))

        # the samples are the computed centers.
        return centers
