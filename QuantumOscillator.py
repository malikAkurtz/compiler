import numpy as np

from Operator import Operator

class QuantumOscillator():
    
    def __init__(self):
        pass
    
    @staticmethod
    def create_ladder_operators(n_cut: int):
        
        a = Operator(matrix=np.zeros((n_cut, n_cut)), basis=n_cut)
        
        for m in range(n_cut):
            for n in range(n_cut):
                if m == (n-1):
                    a.matrix[m][n] = np.sqrt(n)
                    
        a_dagger = Operator(matrix=np.conjugate(a.matrix).T, basis=a.basis)
                    
        return a, a_dagger