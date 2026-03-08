import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Wavefunction import Wavefunction
from constants import PHI_0, hbar

class SFQDriver():
    
    def __init__(self, CC: float, omega_q: float, C: float, EJ: float, EC: float, n_energy: Operator):
        self.theta = CC * PHI_0 * np.sqrt( (2 * omega_q) / C)
        r = (1/2)*( (EJ / (2 * EC)) )**(1/4)
        theta_prime = self.theta / r
        
        self.U_kick = expm( ((-1j * theta_prime) / 2) * n_energy.matrix )
        
    def apply_pulse(self, psi: Wavefunction):
        return psi.apply(self.U_kick)
        
        