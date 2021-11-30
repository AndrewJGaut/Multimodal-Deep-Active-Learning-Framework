from sklearn.cluster import KMeans
from clustering.cluster import *


class KMeans(ClusterMethod):
    def __init__(self, k=10):
        super("KMeans").__init__()
        self.k = k

    def cluster(self, X):
        # first conver the torch tensor into np array
        X = X.cpu().detach().numpy()

        # now, call fit
        kmeans = KMeans(init='random', n_clusters=self.k) # NOTE: init=random here implies that we AREN'T using kmeans++
        labels = kmeans.fit_predict(X)

        # might want to convert this into a torch tensor?
        return labels






