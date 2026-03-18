from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, oscillator: QuantumOscillator, sfq_driver: SFQDriver, initial_state: Wavefunction, basis: str):
        self.oscillator = oscillator
        self.sfq_driver = sfq_driver
        self.state      = initial_state
        
        self.T          = (2*np.pi) / oscillator.angular_frequency # Qubit period [s]
        
        self.basis      = basis

    @staticmethod
    def free_evolve(state: Wavefunction, H0: Operator, T: float, duration: int, basis: str):
        if duration == 1:
            state.apply(operator=Operator({basis: expm(-1j * H0[basis] * (1 * T / 4) / hbar)}))
        elif duration == 2:
            state.apply(operator=Operator({basis: expm(-1j * H0[basis] * (2 * T / 4) / hbar)}))
        elif duration == 3:
            state.apply(operator=Operator({basis: expm(-1j * H0[basis] * (3 * T / 4) / hbar)}))
        elif duration == 4:
            state.apply(operator=Operator({basis: expm(-1j * H0[basis] * (4 * T / 4) / hbar)}))

    def RY(self, theta_target: float):
        
        N = int(np.round(np.abs(theta_target) / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        self.sfq_driver.on_ramp_evolve(self.state)
        
        for _ in range(N):
            self.sfq_driver.apply_pulse(self.state)
            
            System.free_evolve(
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=4,
                basis=self.basis
            )
            
        self.sfq_driver.off_ramp_evolve(self.state)
                        
    def RX(self, theta_target: float):        
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        self.sfq_driver.on_ramp_evolve(self.state)
        
        for _ in range(N):
            
            System.free_evolve(
                state=self.state,
                H0=self.oscillator.H0,
                T=self.T,
                duration=3,
                basis=self.basis
            )
            
            self.sfq_driver.apply_pulse(self.state)
            
            System.free_evolve(
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