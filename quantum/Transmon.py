import numpy as np

from quantum.QuantumOscillator import QuantumOscillator
from quantum.Operator import Operator
from core.constants import *
from quantum.utils import *
        
class Transmon(QuantumOscillator):
    """
    A Transmon system
    """
    def __init__(self, EC: float, EJ_EC: float, n: int):
        self.EC = EC                       # Charging energy
        self.EJ = EJ_EC * self.EC          # Josephson energy
        self.r  = (1/2)*(EJ_EC / 2)**(1/4) # charge zero-point-fluctuation
        
        # Charge operator
        self.n = Operator(
            basis_to_matrix={"charge" : np.diag([n for n in range(int(-(n-1)/2), int((n-1)/2) + 1)])}
        )   
        
        # Kinetic energy operator
        self.T = Operator(
            basis_to_matrix={"charge" : 4*EC*(self.n["charge"]**2)}
        ) 
        
        # Potential energy operator
        self.V = Operator(
            basis_to_matrix={"charge" : create_upper_lower(value=-self.EJ / 2, dim=n)}
        ) 
        
        # Unperturbed Hamiltonian operator
        self.H0 = Operator(
            basis_to_matrix={"charge" : self.T["charge"] + self.V["charge"]}
            ) 
    
        # Diagonalize the Hamiltonian in the charge basis to get energy eigenvalues/eigenvectors
        self.energies, self.energy_states = np.linalg.eigh(self.H0["charge"])
        self.energy_states           = self.energy_states.astype(complex)
    
        # Unperturbed Hamiltonian operator in the energy basis
        self.H0["energy"] = np.diag(self.energies)

        self.anharmonicity           = (self.energies[2] - self.energies[1]) - (self.energies[1] - self.energies[0])
        self.frequency               = (self.energies[1] - self.energies[0]) / h
        self.angular_frequency       = self.frequency * (2*np.pi)
    
        # NOTE: I need to understand this better, Claude gave me this fix and I don't know why it works
        # Fix eigenvector phases so that <psi_i|n|psi_{i+1}> is negative imaginary
        for i in range(len(self.energies) - 1):
            element = self.energy_states[:, i].conj() @ self.n["charge"] @ self.energy_states[:, i + 1]
            if np.abs(element) > 1e-12:
                phase = -1j * element / np.abs(element)
                self.energy_states[:, i + 1] = phase * self.energy_states[:, i + 1]
            
        # Get n in the energy basis
        self.n["energy"] = self.energy_states.conj().T @ self.n["charge"] @ self.energy_states
        
        self.annihilation, self.creation     = QuantumOscillator.create_ladder_operators(n_cut=n)
                
        C = e**2 / (2 * self.EC)
        self.flux                               = Operator(
            basis_to_matrix={"fock" : np.sqrt(hbar / (2*C*self.angular_frequency)) * (self.creation["fock"] + self.annihilation["fock"])}
        )
        
        self.fluxes, self.flux_states = np.linalg.eigh(self.flux["fock"])
        
        self.annihilation["energy"] = self.annihilation["fock"]
        self.creation["energy"]     = self.annihilation["fock"]
        
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
            "T" : self.T,
            "V" : self.V,
            "H0": self.H0,
        }
                       

    
    
    