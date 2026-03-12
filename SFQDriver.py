import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Wavefunction import Wavefunction
from constants import PHI_0, hbar

class SFQDriver():
    
    def __init__(self, CC: float, omega_q: float, C: float, EJ: float, EC: float, n: Operator, fock_approximation: bool):
        self.theta = CC * PHI_0 * np.sqrt( (2 * omega_q) / C)
        
        self.U_kick = Operator(
            basis_to_matrix={"energy" : expm( ((-1j * self.theta) / 2) * n.get_projection(basis="energy") )}
            )
        
        if fock_approximation:
            r = (1/2) * (EJ/(2*EC))**(1/4)
            theta_prime = self.theta / r
            
            self.U_kick = Operator(
                basis_to_matrix={"fock" : expm( ((-1j * theta_prime) / 2) * n.get_projection(basis="fock") )}
                )


                               
    def apply_pulse(self, psi: Wavefunction):
        psi_new = psi.apply(operator=self.U_kick)
        return psi_new
        
        