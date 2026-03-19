import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=5, suppress=True)
from QuantumOscillator import QuantumOscillator
from System import System
from Operator import Operator
from Wavefunction import Wavefunction

class SFQDriver():
    
    def __init__(self, clock_multiplier: int, theta: float, oscillator: QuantumOscillator, basis: str, ramp: list[str], n_proj: int):
        self.theta = theta
        self.ramp = ramp
        self.oscillator = oscillator
        self.basis = basis
        self.clock_multiplier = clock_multiplier
        self.n_proj = n_proj
        
        if basis == "energy":
            theta_prime = oscillator.theta_prime(theta) 
            
            self.U_kick = Operator(
            basis_to_matrix={basis : expm( (-1j * (theta_prime) / 2) * oscillator.n["fock"] )}
            )
        else:       
            self.U_kick = Operator(
            basis_to_matrix={basis : expm( (theta/2) * (oscillator.creation["fock"] - oscillator.annihilation["fock"]) )}
            )           
    
    def apply_pulse(self, state: Wavefunction):
        state.apply(operator=self.U_kick)
    
    def on_ramp_evolve(self, state: Wavefunction):
        
        for sequence in self.ramp:
            for action in sequence:
                if action == "0":
                    # free evolve
                    System.free_evolve(
                        state=state, 
                        H0=self.oscillator.H0,
                        T=(2*np.pi) / self.oscillator.angular_frequency,
                        duration=1,
                        clock_multiplier=self.clock_multiplier,
                        basis=self.basis
                    )
                else:
                    # kick
                    self.apply_pulse(state)
    
    def off_ramp_evolve(self, state: Wavefunction):
        flipped_ramps = [SFQDriver.flip_X_ramp(r) for r in self.ramp][::-1]
        
        for sequence in flipped_ramps:
            for action in sequence:
                if action == "0":
                    # free evolve
                    System.free_evolve(
                        state=state, 
                        H0=self.oscillator.H0,
                        T=(2*np.pi) / self.oscillator.angular_frequency,
                        duration=1,
                        clock_multiplier=self.clock_multiplier,
                        basis=self.basis
                    )
                else:
                    # kick
                    self.apply_pulse(state)
                    
                    System.free_evolve(
                        state=state, 
                        H0=self.oscillator.H0,
                        T=(2*np.pi) / self.oscillator.angular_frequency,
                        duration=1,
                        clock_multiplier=self.clock_multiplier,
                        basis=self.basis
                    )
        
    @staticmethod
    def flip_X_ramp(ramp: str):
        n = len(ramp)
        flipped = ['0'] * n
        for i in range(n):
            if ramp[i] == '1':
                new_pos = (n - i) % n
                flipped[new_pos] = '1'
        return ''.join(flipped)
        
        