import numpy as np

from quantum.QuantumOscillator import QuantumOscillator
from quantum.Operator import Operator
from core.constants import hbar
from quantum.utils import *
        
class HarmonicOscillator(QuantumOscillator):
    """
    A Harmonic Oscillator is a type of System
    Important notes:
    1) In a harmonic oscillator, the fock basis and the energy basis are the same
    2) a_dagger @ a = n = N, the number operator
    """
    def __init__(self, capacitance: float, inductance: float, n_cut: int):
        self.capatiance                      = capacitance
        self.angular_frequency               = angular_frequency = np.sqrt((1 / inductance) / capacitance) # [rad/s]
        self.annihilation, self.creation     = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        
        self.n                               = Operator(
            basis_to_matrix={"fock" : self.creation["fock"] @ self.annihilation["fock"],}
        )
        
        # Hamiltonian opertator in the fock basis
        self.H0                              = Operator(
            basis_to_matrix={"fock": hbar*angular_frequency*(self.n["fock"]+(0.5*np.eye(n_cut)))}
        )
        
        self.energies, self.energy_states    = np.linalg.eigh(self.H0["fock"])
        
        self.n["energy"]            = self.energy_states.conj().T @ self.n["fock"] @ self.energy_states
        self.H0["energy"]           = np.diag(self.energies)
        self.creation["energy"]     = self.energy_states.conj().T @ self.creation["fock"] @ self.energy_states
        self.annihilation["energy"] = self.energy_states.conj().T @ self.annihilation["fock"] @ self.energy_states
        
        self.flux                               = Operator(
            basis_to_matrix={"fock" : np.sqrt(hbar / (2*capacitance*angular_frequency)) * (self.creation["fock"] + self.annihilation["fock"])}
        )
        
        self.fluxes, self.flux_states = np.linalg.eigh(self.flux["fock"])
        
        self.annihilation["flux"] = matrix_change_basis(
                                        transformation_matrix=self.flux_states, 
                                        matrix=self.annihilation["fock"]
                                        )
        self.creation["flux"]     = matrix_change_basis(
                                        transformation_matrix=self.flux_states, 
                                        matrix=self.creation["fock"]
                                        )
        
        self.operators = {
            "n" : self.n,
            "H0": self.H0,
        }