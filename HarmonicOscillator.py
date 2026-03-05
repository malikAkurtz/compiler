import numpy as np

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from constants import hbar   
        
class HarmonicOscillator(QuantumOscillator):
    """
    A Harmonic Oscillator is a type of System
    """
    def __init__(self, mass: float, omega: float, n_cut: int):
        self.mass               = mass
        self.omega              = omega
        self.a, self.a_dagger   = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        self.n                  = Operator(matrix=(self.a_dagger.matrix @ self.a.matrix), basis="fock")
        self.H                  = Operator(matrix=hbar*omega*(self.n.matrix+(0.5*np.eye(n_cut))), basis="fock")
    