import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=5, suppress=True)
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
            basis_to_matrix={basis : expm( (1j * (-theta_prime) / 2) * oscillator.n["energy"] )}
            )
        else:
            # A = (oscillator.creation["fock"] - oscillator.annihilation["fock"]).copy()
            
            # eigenvalues, V = np.linalg.eig(A)
            # V_inv = np.linalg.inv(V)
            
            # print("A:")
            # print(A)
            # print("A eigenvalues:")
            # print(eigenvalues)
            # print("A eigenvectors (V):")
            # print(V)
            # print("V inv")
            # print(V_inv)
            
            # self.U_kick = Operator(
            # basis_to_matrix={basis : V @ expm( (theta / 2) * np.diag(eigenvalues) ) @ V_inv}
            # )
                    
            self.U_kick = Operator(
            basis_to_matrix={basis : expm( (theta/2) * (oscillator.creation["fock"] - oscillator.annihilation["fock"]) )}
            )            
                        
    def apply_pulse(self, psi: Wavefunction):
        psi_new = psi.apply(operator=self.U_kick)
        return psi_new
        
        