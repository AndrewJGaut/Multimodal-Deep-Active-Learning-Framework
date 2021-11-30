from sklearn.cluster import KMeans
from clustering.cluster import *


class KMeans(ClusterMethod):
    def __init__(self, n_clusters=10):
        super("KMeans", n_clusters).__init__()

    def cluster(self, X, n_clusters=None):
        # first conver the torch tensor into np array
        X = X.cpu().detach().numpy()

        if n_clusters is None:
            n_clusters = self.n_clusters

        # now, call fit
        kmeans = KMeans(init='random', n_clusters=self.n_clusters) # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = kmeans.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels






