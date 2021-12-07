"""
Define a parent class for all the clustering methods so that they have a consistent API.
"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class ClusterMethod:
    def __init__(self, method_name, n_clusters):
        self.method_name = method_name
        self.n_clusters = n_clusters


    def cluster(self, X, n_clusters=None):
        """

        :param X: (torch tensor of shape [n_examples,n_features])
        :param n_clusters: if None, use self.n_clustres
        :return: (torch tensor of shape [n_examples,1]) a tensor of the cluster label for each example # a tuple of (cluster centers, cluster labels)
        """
        pass

    def convert_to_numpy(self, X):
        return X.cpu().detach().numpy()

    def name(self):
        return self.method_name


class SklearnAgglomerativeCluster(ClusterMethod):
    def __init__(self, n_clusters=10, linkage='average'):
        super().__init__("AgglomerativeClustering", n_clusters)
        self.linkage = linkage

    def cluster(self, X, n_clusters=10):
        # first conver the torch tensor into np array
        #np_X = self.convert_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage=self.linkage)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = model.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels


class SklearnGMM(ClusterMethod):
    def __init__(self, n_clusters=10):
        super().__init__("GMM", n_clusters)

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        #np_X = self.convert_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        gmm = GaussianMixture(
            n_componentsint=self.n_clusters)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = gmm.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels


class SklearnKMeans(ClusterMethod):
    def __init__(self, n_clusters=10):
        super().__init__("KMeans", n_clusters)

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        #np_X = self.convert_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels


