import numpy as np
from scipy.linalg import expm

from QuantumOscillator import QuantumOscillator
from Transmon import Transmon
from Operator import Operator
from Wavefunction import Wavefunction
from constants import PHI_0, hbar

class SFQDriver():
    
    def __init__(self, theta: float, oscillator: QuantumOscillator, basis: str):
        self.theta = theta
        
        if basis == "energy":
            theta_prime = oscillator.theta_prime(theta) 
            
            self.U_kick = Operator(
            basis_to_matrix={basis : expm( ((-1j * theta_prime) / 2) * oscillator.n["energy"] )}
            )
        else:
            self.U_kick = Operator(
            basis_to_matrix={basis : expm( (-theta/2) * (oscillator.creation["fock"] - oscillator.annihilation["fock"]) )}
            )
                        
    def apply_pulse(self, psi: Wavefunction):
        psi_new = psi.apply(operator=self.U_kick)
        return psi_new
        
        