from __future__ import annotations
import numpy as np

from Operator import Operator
class Wavefunction():
    """
    A wavefunction is a vector in a complex vector space,
    along with a specified basis
    """
    
    def __init__(self, basis_to_coefs: dict):
        self.basis_to_coefs = basis_to_coefs
    
    def get_projection(self, basis: str):
        return self.basis_to_coefs[basis]
    
    def set_projection(self, basis: str, coefs: np.ndarray):
        self.basis_to_coefs[basis] = coefs
        
    def get_probabilities(self, basis: str):
        return np.abs(self.basis_to_coefs[basis])**2
            
    def apply(self, operator: Operator):
        new_basis_to_coefs = {}
        
        # Apply the operator to each basis representation of the wavefunction
        for basis, coefs in self.basis_to_coefs.items():
            new_basis_to_coefs[basis] = (operator.get_projection(basis=basis) @ coefs.reshape(-1, 1)).flatten()
        
        return Wavefunction(new_basis_to_coefs)
              