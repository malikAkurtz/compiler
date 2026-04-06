from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from Transmon import Transmon
from Operator import Operator
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, transmons: list[Transmon], EC_matrix: np.ndarray, thetas: np.ndarray, clock_multiplier: int, initial_state: Wavefunction, ramp: list[str], N_kicks: int, flux_schedule: np.ndarray):
        self.transmons     = transmons
        self.EC_matrix     = EC_matrix
        self.thetas        = thetas
        self.M             = clock_multiplier
        self.state         = initial_state  
        self.ramp          = ramp
        self.N_kicks       = N_kicks
        self.flux_schedule = flux_schedule
        
        # ---- Derive Qubit and Clock Periods ----
        self.T_q           = np.array([(2*np.pi) / t.qubit_angular_frequency for t in self.transmons]) # Qubit period [s]
        # We assume our computational qubits share the same angular frequency s.t. the clock period is well-defined
        self.T_c           = self.T_q[0] / self.M                                                      # Clock period [s]
        
        # ---- Derive Kick Operators for Logical Qubits 1 (and 2)
        self.transmon_to_kick = {}
        
        self.transmon_to_kick[0] = Operator(
                basis_to_matrix={"energy": expm(-1j * (thetas[0] / self.transmons[0].r) / 2 * self.transmons[0].n["energy"])}
            )
        
        if len(transmons) > 0:
            self.transmon_to_kick[2] = Operator(
                basis_to_matrix={"energy": expm(-1j * (thetas[1] / self.transmons[2].r) / 2 * self.transmons[2].n["energy"])}
            )
        
        # ---- Derive Coupling Hamiltonain HC ----
        coupling_sum = 0
        for k in range(len(transmons)):
            for l in range(len(transmons)):
                if k == l:
                    continue
                else:
                    coupling_sum += (self.transmons[k].n * self.EC_matrix[k][l] * self.transmons[l].n)
                
        self.HC = Operator(
            basis_to_matrix={"energy" : 4 * coupling_sum}
        )
        
        # ---- Derive the Unperturbed Hamiltonian H0 ----
        self.H0 = Operator(
            basis_to_matrix={"energy" : np.sum([t.H0 for t in self.transmons]) + self.HC["energy"]}
        )
                
    def free_evolve(self, clock_cycles: int):
        self.state.apply(
            operator=Operator(
                {"energy": expm(-1j * self.H0["energy"] * (clock_cycles * self.T_c) / hbar)}
                )
            )

    def RY(self, qubit: int):
        
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