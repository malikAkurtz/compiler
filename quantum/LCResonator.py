import numpy as np

from quantum.QuantumOscillator import QuantumOscillator
from quantum.Operator import Operator
from core.constants import hbar, h, e
from quantum.utils import *
        
class HarmonicOscillator(QuantumOscillator):
    """
    A Harmonic Oscillator is a type of System
    Important notes:
    1) In a harmonic oscillator, the fock basis and the energy basis are the same
    2) a_dagger @ a = F, the number operator
    """
    def __init__(self, capacitance: float, inductance: float, n_cut: int):  
        self.C = capacitance
        self.L = inductance
        
        self.angular_frequency = np.sqrt(1 / (self.C*self.L))    
        self.frequency         = self.angular_frequency * 2 * np.pi
        
        self.annihilation, self.creation     = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        
        self.F                               = Operator(
            basis_to_matrix={"fock" : self.creation["fock"] @ self.annihilation["fock"],}
        )
        
        # Hamiltonian opertator in the fock basis
        self.H0                              = Operator(
            basis_to_matrix={"fock": hbar*self.angular_frequency*(self.F["fock"]+(0.5*np.eye(n_cut)))}
        )
        
        self.energies, self.energy_states    = np.linalg.eigh(self.H0["fock"])
        
        self.anharmonicity           = (self.energies[2] - self.energies[1]) - (self.energies[1] - self.energies[0])
        
        self.F["energy"]            = self.energy_states.conj().T @ self.F["fock"] @ self.energy_states
        self.H0["energy"]           = np.diag(self.energies)
        self.creation["energy"]     = self.energy_states.conj().T @ self.creation["fock"] @ self.energy_states
        self.annihilation["energy"] = self.energy_states.conj().T @ self.annihilation["fock"] @ self.energy_states
        
        self.flux                               = Operator(
            basis_to_matrix={"fock" : np.sqrt(hbar / (2*capacitance*self.angular_frequency)) * (self.creation["fock"] + self.annihilation["fock"])}
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
        
        self.n_zpf = (1 / (2*e)) * np.sqrt(hbar * self.angular_frequency * self.C / 2)
        
        self.n = Operator(
            basis_to_matrix={"energy" : 1j*self.n_zpf*(self.creation["energy"] - self.annihilation["energy"])}
        )
        
        self.operators = {
            "F" : self.F,
            "H0": self.H0,
            "n" : self.n
        }