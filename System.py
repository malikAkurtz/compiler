from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, clock_multiplier: int, oscillator: QuantumOscillator, sfq_driver: SFQDriver, initial_state: Wavefunction, basis: str, N: int):
        self.clock_multiplier = clock_multiplier
        self.oscillator = oscillator
        self.sfq_driver = sfq_driver
        self.state      = initial_state
        self.N          = N
        
        self.T          = (2*np.pi) / oscillator.angular_frequency # Qubit period [s]
        
        self.basis      = basis

    @staticmethod
    def free_evolve(state: Wavefunction, H0: Operator, T: float, duration: int, clock_multiplier: int, basis: str):
        state.apply(operator=Operator({basis: expm(-1j * H0[basis] * (duration * T / clock_multiplier) / hbar)}))

    def RY(self, theta_target: float):
        
        N = self.N
        
        self.sfq_driver.on_ramp_evolve(self.state)
        
        for _ in range(N):
            self.sfq_driver.apply_pulse(self.state)
            
            System.free_evolve(
                clock_multiplier=self.clock_multiplier,
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=self.sfq_driver.clock_multiplier,
                basis=self.basis
            )
            
        self.sfq_driver.off_ramp_evolve(self.state)
                        
    def RX(self, theta_target: float):        
        
        N = self.N
        
        self.sfq_driver.on_ramp_evolve(self.state)
        
        for _ in range(N):
            
            System.free_evolve(
                clock_multiplier=self.clock_multiplier,
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=3,
                basis=self.basis
            )
            
            self.sfq_driver.apply_pulse(self.state)
            
            System.free_evolve(
                clock_multiplier=self.clock_multiplier,
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=1,
                basis=self.basis
            )
            
        self.sfq_driver.off_ramp_evolve(self.state)
    
    def X(self):
        return self.RX(np.pi) # ONLY FOR SINGLE QUBIT CASE
            
    def Hadamard(self):
        # RY(pi/2) rotation
        self.RY(np.pi/2)
        # RZ(pi) rotation
        System.free_evolve(
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=2,
                basis=self.basis
            )