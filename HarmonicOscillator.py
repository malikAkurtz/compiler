import numpy as np

from Operator import Operator
from constants import hbar   
        
class HarmonicOscillator():
    """
    A Harmonic Oscillator is a type of System
    """
    def __init__(self, mass: float, omega: float, n_cut: int):
        self.mass               = mass
        self.omega              = omega
        self.a, self.a_dagger   = HarmonicOscillator.create_ladder_operators(n_cut="fock")
        self.n                  = Operator(matrix=(self.a_dagger.matrix @ self.a.matrix), basis="fock")
        self.H                  = Operator(matrix=hbar*omega*(self.n.matrix+(0.5*np.eye(n_cut))), basis="fock")
    
    @staticmethod
    def create_ladder_operators(n_cut: int):
        
        a = Operator(matrix=np.zeros((n_cut, n_cut)), basis=n_cut)
        
        for m in range(n_cut):
            for n in range(n_cut):
                if m == (n-1):
                    a.matrix[m][n] = np.sqrt(n)
                    
        a_dagger = Operator(matrix=np.conjugate(a.matrix).T, basis=a.basis)
                    
        return a, a_dagger
    
    