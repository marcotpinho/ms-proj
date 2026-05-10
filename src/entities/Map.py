import numpy as np
from scipy.spatial.distance import cdist


class Map:
    def __init__(
            self,
            rpositions: np.ndarray,
            rvalues: np.ndarray,
            predict_distances: bool = False
    ):
        self.rpositions = rpositions
        self.rvalues = rvalues
        self.distmx = cdist(rpositions, rpositions, metric="euclidean")
        if predict_distances:
            max_vals = rpositions.max(axis=0)
            min_vals = rpositions.min(axis=0)
            self.center = (max_vals + min_vals) / 2
            self.diag = np.linalg.norm(max_vals - min_vals) + 1e-8
        else:
            self.center = None
            self.diag = None
