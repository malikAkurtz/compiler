import numpy as np

class Basis():
    """
    A Basis is a set of linearly independent vectors
    that span the entire vector space
    """
    
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors