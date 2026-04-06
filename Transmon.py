import numpy as np
from typing import Optional

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from constants import *   
from utils import *
        
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
        energies, energy_states = np.linalg.eigh(self.H0["charge"])
        energy_states           = energy_states.astype(complex)
    
        # Unperturbed Hamiltonian operator in the energy basis
        self.H0["energy"] = np.diag(energies)

        self.anharmonicity           = (energies[2] - energies[1]) - (energies[1] - energies[0])
        self.qubit_frequency         = (energies[1] - energies[0]) / h 
        self.qubit_angular_frequency = self.qubit_frequency * (2*np.pi)
    
        # NOTE: I need to understand this better, Claude gave me this fix and I don't know why it works
        # Fix eigenvector phases so that <psi_i|n|psi_{i+1}> is negative imaginary
        for i in range(len(energies) - 1):
            element = energy_states[:, i].conj() @ self.n["charge"] @ energy_states[:, i + 1]
            if np.abs(element) > 1e-12:
                phase = -1j * element / np.abs(element)
                energy_states[:, i + 1] = phase * energy_states[:, i + 1]
            
        # Get n in the energy basis
        self.n["energy"] = energy_states.conj().T @ self.n["charge"] @ energy_states
        
        self.operators = {
            "n" : self.n,
            "T" : self.T,
            "V" : self.V,
            "H0": self.H0,
        }
                        
        
            
        
    
    
    