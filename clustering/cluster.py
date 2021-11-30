"""
Define a parent class for all the clustering methods so that they have a consistent API.
"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import GaussianMixture
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

    def conver_to_numpy(self, X):
        return X.cpu().detach().numpy()


class AgglomerativeCluster(ClusterMethod):
    def __init__(self, n_clusters=10, linkage='average'):
        super("GMM", n_clusters).__init__()
        self.linkage = linkage

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        np_X = self.conver_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        model = AgglomerativeClustering(n_componentsint=self.n_clusters,
                                        linkage=self.linkage)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = model.fit_predict(np_X)

        # might want to convert this into a torch tensor?
        return labels


class GMM(ClusterMethod):
    def __init__(self, n_clusters=10):
        super("GMM", n_clusters).__init__()

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        np_X = self.conver_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        gmm = GaussianMixture(
            n_componentsint=self.n_clusters)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = gmm.fit_predict(np_X)

        # might want to convert this into a torch tensor?
        return labels


class KMeansPlusPlus(ClusterMethod):
    def __init__(self, n_clusters=10):
        super("KMeans++", n_clusters).__init__()

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        X = X.cpu().detach().numpy()

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        kmeans = KMeans(init='k-means++',
                        n_clusters=self.n_clusters)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = kmeans.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels


class KMeans(ClusterMethod):
    def __init__(self, n_clusters=10):
        super("KMeans", n_clusters).__init__()

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        X = X.cpu().detach().numpy()

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        kmeans = KMeans(init='random',
                        n_clusters=self.n_clusters)  # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = kmeans.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels


