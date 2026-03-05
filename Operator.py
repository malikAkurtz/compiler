import numpy as np

class Operator():
    """
    An Operator is a matrix in a chosen basis
    """
    
    def __init__(self, matrix: np.ndarray, basis: str):
        self.matrix = matrix
        self.basis  = basis

    def __str__(self):
        return f"Matrix: {self.matrix} \n Basis: {self.basis}"
    