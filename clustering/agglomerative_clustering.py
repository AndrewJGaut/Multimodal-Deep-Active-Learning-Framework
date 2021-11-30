from sklearn.cluster import AgglomerativeClustering
from clustering.cluster import *


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
        model = AgglomerativeClustering(n_componentsint=self.n_clusters, linkage=self.linkage) # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = model.fit_predict(np_X)

        # might want to convert this into a torch tensor?
        return labels


