from sklearn.cluster import GaussianMixture
from clustering.cluster import *


class GMM(ClusterMethod):
    def __init__(self, n_clusters=10):
        super("GMM", n_clusters).__init__()

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        np_X = self.conver_to_numpy(X)

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        gmm = GaussianMixture(n_componentsint=self.n_clusters) # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = gmm.fit_predict(np_X)

        # might want to convert this into a torch tensor?
        return labels





