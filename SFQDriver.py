import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Wavefunction import Wavefunction
from constants import PHI_0, hbar

class SFQDriver():
    
    def __init__(self, CC: float, omega_q: float, C: float, H0: Operator, a: Operator, a_dagger: Operator):
        self.theta = CC * PHI_0 * np.sqrt( (2 * omega_q) / C)
        self.U_kick = expm((-self.theta/2) * (a_dagger.matrix - a.matrix))        
        
    def apply_pulse(self, psi: Wavefunction):
        return psi.apply(self.U_kick)
        
        