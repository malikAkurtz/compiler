import numpy as np

class Operator():
    """
    An Operator is a matrix in a chosen basis
    """
    
    def __init__(self, basis_to_matrix: dict):
        self.basis_to_matrix = basis_to_matrix
    
    def get_projection(self, basis: str):
        return self.basis_to_matrix[basis]
    
    def __getitem__(self, key):
        return self.get_projection(basis=key)

    def set_projection(self, basis: str, matrix: np.ndarray):
        self.basis_to_matrix[basis] = matrix
        
    def __setitem__(self, key, value):
        self.set_projection(basis=key, matrix=value)
    
    def __delitem__(self, key):
        del self.basis_to_matrix[key]
    