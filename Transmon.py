import numpy as np

from Operator import Operator
from constants import *   
from utils import *
        
class Transmon():
    """
    A Transmon system
    """
    def __init__(self, EC: float, EJ_EC: float, n_cut: int):
        self.EC                           = EC # Charging energy
        self.EJ                           = EJ_EC * EC # Josephson energy
        self.n                            = Operator(matrix=np.diag([n for n in range(int(-(n_cut-1)/2), int((n_cut-1)/2) + 1)]), basis="charge") # Charge operator (charge basis)
        self.T                            = Operator(matrix=4*EC*(self.n.matrix**2), basis=self.n.basis) # Kinetic energy operator (charge basis)
        self.V                            = Operator(matrix=create_upper_lower(value=-self.EJ / 2, dim=n_cut), basis=self.n.basis) # Potential energy operator (charge basis)
        self.H                            = Operator(matrix=self.T.matrix + self.V.matrix, basis=self.n.basis) # Hamiltonian operator (charge basis)
        self.energies, self.energy_states = np.linalg.eigh(self.H.matrix) # Eigenvalues + eigenvectors of Hamiltonian operator (charge basis)
        self.H0                           = Operator(matrix=np.diag(self.energies), basis="energy")
        self.alpha                        = (self.energies[2] - self.energies[1]) - (self.energies[1] - self.energies[0])  # Anharmonicity
        self.fq                           = (self.energies[1] - self.energies[0]) / h 
    
    