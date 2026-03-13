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
    def __init__(self, charging_energy: float, EJ_EC: float, n_cut: int, basis: str):
        self.EC                           = charging_energy
        self.EJ                           = EJ_EC * self.EC # Josephson energy
        self.r                            = (1/2)*(EJ_EC / 2)**(1/4) # charge zero-point-fluctuation
        
        # Charge operator
        self.n                            = Operator(
            basis_to_matrix={"charge" : np.diag([n for n in range(int(-(n_cut-1)/2), int((n_cut-1)/2) + 1)])}
        )   
        
        # Kinetic energy operator
        self.T                            = Operator(
            basis_to_matrix={"charge" : 4*self.EC*(self.n["charge"]**2)}
        ) 
        
        # Potential energy operator (charge basis)
        self.V                            = Operator(
            basis_to_matrix={"charge" : create_upper_lower(value=-self.EJ / 2, dim=n_cut)}
        ) 
        
        # Hamiltonian operator (charge basis)
        self.H0                           = Operator(
            basis_to_matrix={"charge" : self.T["charge"] + self.V["charge"]}
            ) 
        
        self.energies, self.energy_states = np.linalg.eigh(self.H0["charge"]) # Eigenvalues + eigenvectors of Hamiltonian operator (charge basis)
        
        self.H0["energy"] = np.diag(self.energies)

        # Number operator
        self.N                            = Operator(
            basis_to_matrix={"energy": np.diag(np.arange(n_cut))}
        )
        
        self.n["energy"] = matrix_change_basis(
                            transformation_matrix=self.energy_states,
                            matrix=self.n["charge"]
                            )
    
        self.anharmonicity          = (self.energies[2] - self.energies[1]) - (self.energies[1] - self.energies[0])
        self.frequency              = (self.energies[1] - self.energies[0]) / h 
        self.angular_frequency      = self.frequency * (2*np.pi)
        
        if basis == "fock":  
            self.annihilation, self.creation  = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
            self.H0["fock"] = hbar * self.angular_frequency * \
                        ( (self.creation["fock"] @ self.annihilation["fock"]) ) \
                        - ( (self.anharmonicity / 2) * (self.creation["fock"] @ self.creation["fock"]) \
                            @ (self.annihilation["fock"] @ self.annihilation["fock"]))

            self.n["fock"] = 1j * self.r * (self.creation["fock"] - self.annihilation["fock"])
    
    def theta_prime(self, theta):
        return theta / self.r
                        
        
            
        
    
    
    