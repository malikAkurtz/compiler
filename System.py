from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, energy_states: np.ndarray, n: Operator, n_zpf: float, theta: float, H0: Operator, qubit_angular_frequency: float, clock_multiplier: int, initial_state: Wavefunction, N: int, ramp: list[str]):
        self.theta   = theta
        self.H0      = H0
        self.omega_q = qubit_angular_frequency
        self.M       = clock_multiplier
        self.state   = initial_state
        self.N       = N
        self.ramp    = ramp
        
        self.T_q     = (2*np.pi) / qubit_angular_frequency # Qubit period [s]
        self.T_c     = self.T_q / self.M                   # Clock period [s]
                
        self.U_kick = Operator(
            basis_to_matrix={"energy": expm(-1j * (theta / n_zpf) / 2 * n["energy"])}
        )
                
        
    def free_evolve(self, clock_cycles: int):
        self.state.apply(
            operator=Operator(
                {"energy": expm(-1j * self.H0["energy"] * (clock_cycles * self.T_c) / hbar)}
                )
            )

    def RY(self):
        
        N = self.N
        
        self.on_ramp_evolve()
                
        for _ in range(N):
            self.state.apply(self.U_kick)
            
            self.free_evolve(
                clock_cycles=self.M,
            )
        
        self.off_ramp_evolve()
            
    def on_ramp_evolve(self):
    
        for sequence in self.ramp:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.free_evolve(clock_cycles=1)
                else:
                    # kick
                    self.state.apply(self.U_kick)

                    # free evolve for a single clock cycle
                    self.free_evolve(clock_cycles=1)

    def off_ramp_evolve(self):
        flipped_ramps = [self.flip_X_ramp(r) for r in self.ramp][::-1]
        
        for sequence in flipped_ramps:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.free_evolve(clock_cycles=1)
                else:
                    # kick
                    self.state.apply(self.U_kick)

                    # free evolve for a single clock cycle
                    self.free_evolve(clock_cycles=1)

    @staticmethod
    def flip_X_ramp(ramp: str):
        n = len(ramp)
        flipped = ['0'] * n
        for i in range(n):
            if ramp[i] == '1':
                new_pos = (n - i) % n
                flipped[new_pos] = '1'
        return ''.join(flipped)