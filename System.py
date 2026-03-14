import numpy as np
from scipy.linalg import expm

from QuantumOscillator import QuantumOscillator
from Operator import Operator
from Transmon import Transmon
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, oscillator: QuantumOscillator, sfq_driver: SFQDriver, initial_state: Wavefunction, basis: str):
        self.oscillator = oscillator
        self.sfq_driver = sfq_driver
        self.state      = initial_state
                
        self.T          = (2*np.pi) / self.oscillator.angular_frequency # Qubit period [s]
        
        self.basis      = basis
        
    
        self.U_free_1 = Operator({self.basis: expm(-1j * self.oscillator.H0[self.basis] * (1 * self.T / 4) / hbar)})
        self.U_free_2 = Operator({self.basis: expm(-1j * self.oscillator.H0[self.basis] * (2 * self.T / 4) / hbar)})
        self.U_free_3 = Operator({self.basis: expm(-1j * self.oscillator.H0[self.basis] * (3 * self.T / 4) / hbar)})
        self.U_free_4 = Operator({self.basis: expm(-1j * self.oscillator.H0[self.basis] * (4 * self.T / 4) / hbar)})
    
        
    def free_evolve(self, duration: int):
        if duration == 1:
            self.state = self.state.apply(operator=self.U_free_1)
        elif duration == 2:
            self.state = self.state.apply(operator=self.U_free_2)
        elif duration == 3:
            self.state = self.state.apply(operator=self.U_free_3)
        elif duration == 4:
            self.state = self.state.apply(operator=self.U_free_4)

    def RY(self, theta_target: float):
        U_target = np.array([
            [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
            [np.sin(theta_target / 2), np.cos(theta_target / 2)]
        ])
        
        N = int(np.round(np.abs(theta_target) / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = Operator({self.basis: np.eye(N=len(self.state[self.basis]))})
        for _ in range(N):
            self.state = self.sfq_driver.apply_pulse(self.state)
            U[self.basis] = self.sfq_driver.U_kick[self.basis] @ U[self.basis]
            
            self.free_evolve(duration=4)
            U[self.basis] = self.U_free_4[self.basis] @ U[self.basis]
        
        return U, U_target
    
    def RX(self, theta_target: float):        
        U_target = np.array([
            [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
            [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
        ])
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = Operator({self.basis: np.eye(N=len(self.state[self.basis]))})
        for _ in range(N):
            self.free_evolve(duration=3)
            U[self.basis] = self.U_free_3[self.basis] @ U[self.basis]
            
            self.state = self.sfq_driver.apply_pulse(self.state)
            U[self.basis] = self.sfq_driver.U_kick[self.basis] @ U[self.basis]
            
            self.free_evolve(duration=1)
            U[self.basis] = self.U_free_1[self.basis] @ U[self.basis]
            
        return U, U_target
    
    def X(self):
        return self.RX(theta_target=np.pi) # ONLY FOR SINGLE QUBIT CASE
            
    def Hadamard(self):
        U_target = np.array([
            [1.0, 1.0],
            [1.0, -1.0]
        ])
        U_target *= 1 / np.sqrt(2)
        
        U = Operator({self.basis: np.eye(N=len(self.state[self.basis]))})
        # RY(pi/2) rotation
        RY, _ = self.RY(np.pi/2)
        U = RY[self.basis] @ U[self.basis]
        # RZ(pi) rotation
        self.free_evolve(duration=2)
        U = self.U_free_2[self.basis] @ U[self.basis]
        
        return U, U_target

        