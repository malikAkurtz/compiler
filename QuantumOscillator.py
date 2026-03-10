import numpy as np

from Operator import Operator

class QuantumOscillator():
    
    def __init__(self):
        pass
    
    @staticmethod
    def create_ladder_operators(n_cut: int):
        fock_annihilation_matrix = np.zeros((n_cut, n_cut))
        
        for m in range(n_cut):
            for n in range(n_cut):
                if m == (n-1):
                    fock_annihilation_matrix[m][n] = np.sqrt(n)
                    
        annihilation = Operator(
                        basis="fock",
                        matrix=fock_annihilation_matrix
                        )
                    
        creation = Operator(
                    basis="fock",
                    matrix=fock_annihilation_matrix.conj().T, 
                    )
                    
        return annihilation, creation