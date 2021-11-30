"""
Define a parent class for all the clustering methods so that they have a consistent API.



        :param k: (int) the number of clusters to compute
"""



class ClusteringMethod:
    def __init__(self, method_name):
        self.method_name = method_name

    def cluster(self, X):
        """

        :param X: (torch tensor of shape [n_examples,n_features])
        :return: (torch tensor of shape [n_examples,1]) a tensor of the cluster label for each example # a tuple of (cluster centers, cluster labels)
        """
        pass


