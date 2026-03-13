import numpy as np

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from constants import hbar   
from utils import *
        
class HarmonicOscillator(QuantumOscillator):
    """
    A Harmonic Oscillator is a type of System
    Important notes:
    1) In a harmonic oscillator, the fock basis and the energy basis are the same
    2) a_dagger @ a = n = N, the number operator
    """
    def __init__(self, mass: float, angular_frequency: float, n_cut: int):
        self.mass                            = mass
        self.angular_frequency               = angular_frequency
        self.annihilation, self.creation     = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        
        self.n                               = Operator(
            basis_to_matrix={"fock" : self.creation["fock"] @ self.annihilation["fock"]}
        )
        
        # Number operator (=n for a harmonic oscillator, just here for convention)
        self.N                            = Operator(
            basis_to_matrix={"fock": np.diag(np.arange(n_cut, dtype=float))}
        )
        
        # Hamiltonian opertator in the fock basis
        self.H0                              = Operator(
            basis_to_matrix={"fock": hbar*angular_frequency*(self.n["fock"]+(0.5*np.eye(n_cut)))}
        )
        
        self.energies, self.energy_states    = np.linalg.eigh(self.H0["fock"])
        
        self.x                               = Operator(
            basis_to_matrix={"fock" : np.sqrt(hbar / (2*mass*angular_frequency)) * (self.creation["fock"] + self.annihilation["fock"])}
        )
        
        self.positions, self.position_states = np.linalg.eigh(self.x["fock"])
        
        self.annihilation["position"] = matrix_change_basis(
                                        transformation_matrix=self.position_states, 
                                        matrix=self.annihilation["fock"]
                                        )
        self.creation["position"]     = matrix_change_basis(
                                        transformation_matrix=self.position_states, 
                                        matrix=self.creation["fock"]
                                        )
