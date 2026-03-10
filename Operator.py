import numpy as np

class Operator():
    """
    An Operator is a matrix in a chosen basis
    """
    
    def __init__(self, basis: np.ndarray, matrix: str, ):
        self.basis_to_matrix = {basis : matrix}

    def add_projection(self, basis: str, matrix: np.ndarray):
        self.basis_to_matrix[basis] = matrix
        
    def get_projection(self, basis: str):
        return self.basis_to_matrix[basis]
    
    