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
    def __init__(self, EC: float, EJ_EC: float, n_cut: int, dim_sub: Optional[int] | None):
        self.EC                           = EC # Charging energy
        self.EJ                           = EJ_EC * EC # Josephson energy
        
        # Charge operator
        self.n                            = Operator(
            basis_to_matrix={"charge" : np.diag([n for n in range(int(-(n_cut-1)/2), int((n_cut-1)/2) + 1)])}
        )
        
        # Kinetic energy operator
        self.T                            = Operator(
            basis_to_matrix={"charge" : 4*EC*(self.n.get_projection("charge")**2)}
        ) 
        
        self.V                            = Operator(
            basis_to_matrix={"charge" : create_upper_lower(value=-self.EJ / 2, dim=n_cut)}
        ) # Potential energy operator (charge basis)
        
        self.H                            = Operator(
            basis_to_matrix={"charge" : self.T.get_projection("charge") + self.V.get_projection("charge")}
            ) # Hamiltonian operator (charge basis)
        
        self.energies, self.energy_states = np.linalg.eigh(self.H.get_projection("charge")) # Eigenvalues + eigenvectors of Hamiltonian operator (charge basis)
        
        self.H0                           = Operator(
        basis_to_matrix={"energy" : np.diag(self.energies)}
        )
        
        # Number operator
        self.N                            = Operator(
            basis_to_matrix={"energy": np.diag(np.arange(n_cut))}
        )
        
        self.n.set_projection(
            basis="energy",
            matrix=matrix_change_basis(
                transformation_matrix=self.energy_states,
                matrix=self.n.get_projection("charge")
                )
            )
    
        self.anharmonicity                = (self.energies[2] - self.energies[1]) - (self.energies[1] - self.energies[0])
        self.qubit_frequency              = (self.energies[1] - self.energies[0]) / h 
        self.qubit_angular_frequency      = self.qubit_frequency * (2*np.pi)
        
        # For Fock approximation
        self.annihilation, self.creation  = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
        
        if dim_sub != None:
            self.n.set_projection(
                basis="energy",
                matrix=self.n.get_projection("energy")[:dim_sub, :dim_sub]
            )
            self.H0.set_projection(
                basis="energy",
                matrix=self.H0.get_projection("energy")[:dim_sub, :dim_sub] 
            )
            self.N.set_projection(
                basis="energy",
                matrix=self.N.get_projection("energy")[:dim_sub, :dim_sub] 
            )
    
    
    