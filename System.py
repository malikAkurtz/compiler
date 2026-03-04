import numpy as np

from Operator import Operator
from Basis import Basis
from constants import hbar

class System():
    """
    A System object is a projected Hamiltonian H
    along with a projected wavefunction psi in an initial state
    """
    
    def __init__(self, H: Operator, psi: np.ndarray, observables: list[Operator]):
        self.H           = H
        self.state       = psi
        self.observables = observables
        
        
class HarmonicOscillator():
    """
    A Harmonic Oscillator is a type of System
    """
    def __init__(self, mass: float, omega: float, n_cut: int):
        self.mass               = mass
        self.omega              = omega
        self.a, self.a_dagger   = HarmonicOscillator.create_ladder_operators(n_cut=n_cut)
        self.n                  = Operator(matrix=(self.a_dagger.matrix @ self.a.matrix), basis=self.a.basis)
        self.H                  = Operator(matrix=hbar*omega*(self.n.matrix+(0.5*np.eye(n_cut))), basis=self.a.basis)
    
    @staticmethod
    def create_ladder_operators(n_cut: int):
        unit_vectors = []
        for i in range(n_cut):
            unit_vector = n_cut * [0]
            unit_vector[i] = 1
            unit_vectors.append(unit_vector)
        unit_vectors = np.array(unit_vectors)
        
        a = Operator(matrix=np.zeros((n_cut, n_cut)), basis=Basis(vectors=unit_vectors))
        
        for m in range(n_cut):
            for n in range(n_cut):
                if m == (n-1):
                    a.matrix[m][n] = np.sqrt(n)
                    
        a_dagger = Operator(matrix=np.conjugate(a.matrix).T, basis=a.basis)
                    
        return a, a_dagger
    
    