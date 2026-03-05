import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Wavefunction import Wavefunction

class SFQPulse():
    
    def __init__(self, theta: float, a: Operator, a_dagger: Operator):
        self.U_kick = np.expm((-theta/2) * (a_dagger-a))
        
    def apply_pulse(self, psi: Wavefunction):
        psi_new = psi.apply(self.U_kick)
        