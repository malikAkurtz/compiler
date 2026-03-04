import numpy as np

from Basis import Basis

class Operator():
    """
    An Operator is a matrix in a chosen basis
    """
    
    def __init__(self, matrix: np.ndarray, basis: Basis):
        self.matrix = matrix
        self.basis  = basis
        
    def __str__(self):
        return f"{self.matrix}"
    