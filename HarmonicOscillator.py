import numpy as np

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from constants import hbar   
        
class HarmonicOscillator(QuantumOscillator):
    """
    A Harmonic Oscillator is a type of System
    """
    def __init__(self, mass: float, angular_frequency: float, n_cut: int):
        self.mass                            = mass
        self.angular_frequency               = angular_frequency
        self.annihilation, self.creation     = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        self.n                               = Operator(
                                                matrix=(self.creation.matrix @ self.annihilation.matrix), 
                                                basis="fock"
                                                )
        self.H                               = Operator(
                                                matrix=hbar*angular_frequency*(self.n.matrix+(0.5*np.eye(n_cut))), 
                                                basis="fock"
                                                )
        self.energies, self.energy_states    = np.linalg.eigh(self.H.matrix)
        self.x                               = Operator(
                                                matrix=np.sqrt(hbar / (2*mass*angular_frequency)) * (self.creation.matrix + self.annihilation.matrix),
                                                basis="fock"
                                                )
        self.positions, self.position_states = np.linalg.eigh(self.x.matrix)
