from __future__ import annotations
import numpy as np
from scipy.linalg import expm
import copy

from Transmon import Transmon
from Operator import Operator
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, transmons: list[Transmon], EC_matrix: np.ndarray, thetas: np.ndarray, clock_multiplier: int, initial_state: Wavefunction, ramp: list[str], N_kicks: int):
        self.transmons     = transmons
        self.EC_matrix     = EC_matrix
        self.thetas        = thetas
        self.M             = clock_multiplier
        self.state         = initial_state  
        self.ramp          = ramp
        self.N_kicks       = N_kicks
        self.coupling      = False
        
        self.n_full        = len(self.state["energy"]) # Dimension of full Hilbert space of the system
        self.n_trunc       = int(round(self.n_full**(1/len(self.transmons))))
        
        # ---- Create Projectors for each individual qubit subspace ----
        
        # self.projectors = [
        #     System.create_projector(
        #         bases=["energy"], 
        #         n_trunc=self.n_trunc,
        #         ind=k,
        #         num_subsystems=len(self.transmons)
        #         )
        #     for k in range(len(self.transmons))
        #     ]
        
        
        # ---- Truncate each transmon to n_trunc x n_trunc, then upgrade it to the n_full state space
        print("Starting truncate/upgrade...")
        self.sys_transmons = self.sys_transmons = copy.deepcopy(self.transmons)
        
        for k in range(len(self.transmons)):
            sys_transmon = self.sys_transmons[k]
            
            for label in list(sys_transmon.operators.keys()):
                
                sys_transmon.operators[label] = System.upgrade(
                    operator=System.truncate(sys_transmon.operators[label], self.n_trunc),
                    n_trunc=self.n_trunc,
                    idx=k,
                    num_subsystems=len(self.transmons)
                )
        
        # ---- Derive Qubit and Clock Periods ----
        self.T_q           = np.array([(2*np.pi) / t.qubit_angular_frequency for t in self.transmons]) # Qubit period [s]
        # We assume our computational qubits share the same angular frequency s.t. the clock period is well-defined
        self.T_c           = self.T_q[0] / self.M                                                      # Clock period [s]
        
        # ---- Derive Kick Operators for Logical Qubits 1 (and 2)
        print("Building kick operators...")
        self.transmon_to_kick = {}
        
        self.transmon_to_kick[0] = Operator(
                    basis_to_matrix={"energy": expm(-1j * (thetas[0] / self.sys_transmons[0].r) / 2 * self.transmons[0].n["energy"])}
                )
        self.transmon_to_kick[0] = System.truncate(
            operator=self.transmon_to_kick[0],
            n_trunc=self.n_trunc
        )
        self.transmon_to_kick[0] = System.upgrade(
            operator=self.transmon_to_kick[0],
            n_trunc=self.n_trunc,
            idx=0,
            num_subsystems=len(self.transmons)
        )
        
        if len(self.transmons) > 1:
            self.transmon_to_kick[2] = Operator(
                        basis_to_matrix={"energy": expm(-1j * (thetas[1] / self.transmons[2].r) / 2 * self.transmons[2].n["energy"])}
                    )
            self.transmon_to_kick[2] = System.truncate(
                operator=self.transmon_to_kick[2],
                n_trunc=self.n_trunc
            )
            self.transmon_to_kick[2] = System.upgrade(
                operator=self.transmon_to_kick[2],
                n_trunc=self.n_trunc,
                idx=2,
                num_subsystems=len(self.transmons)
            )
        
        # ---- Derive Coupling Hamiltonain HC ----
        print("Building coupling Hamiltonian...")
        coupling_sum = 0
        for k in range(len(self.transmons)):
            for l in range(len(self.transmons)):
                if k == l:
                    continue
                else:   
                    coupling_sum += (self.EC_matrix[k][l] * self.sys_transmons[k].operators["n"]["energy"] @ self.sys_transmons[l].operators["n"]["energy"])
                
        self.HC = Operator(
            basis_to_matrix={"energy" : 4 * coupling_sum}
        )
        
        # ---- Derive the Unperturbed Hamiltonian H0 ----
        print("Building H0...")
        total_h0 = self.sys_transmons[0].operators["H0"]["energy"]
        for t in self.sys_transmons[1:]:
            total_h0 = total_h0 + t.operators["H0"]["energy"]

        self.H0 = Operator(
            basis_to_matrix={"energy": total_h0 + self.HC["energy"]}
        )
        
        print(f"n_full = {self.n_full}")
        print(f"n_trunc = {self.n_trunc}")
        print(f"H0 shape = {self.H0['energy'].shape}")
        print(f"state length = {len(self.state['energy'])}")
        
        # ---- For single qubit case ----
        print("Computing U1, UM...")
        self.U1 = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (1 * self.T_c) / hbar)}
        )
        
        self.UM = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (self.M * self.T_c) / hbar)}
        )
        
        print("System init complete.")
                
    def switch_coupling(self):
        if self.coupling == False:
            self.coupling == True
        else:
            self.coupling == False
        
    def free_evolve(self, clock_cycles: int):
        self.state.apply(
            operator=Operator(
                {"energy": expm(-1j * self.H0["energy"] * (clock_cycles * self.T_c) / hbar)}
                )
            )

    def RY(self, k: int):
        
        self.on_ramp_evolve(k)
                
        for _ in range(self.N_kicks):
            self.state.apply(self.transmon_to_kick[k])
            
            self.state.apply(self.UM)
            # self.free_evolve(
            #     clock_cycles=self.M,
            # )
        
        self.off_ramp_evolve(k)
            
    def on_ramp_evolve(self, k: int):
    
        for sequence in self.ramp:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.state.apply(self.U1)
                else:
                    # kick
                    self.state.apply(self.transmon_to_kick[k])

                    # free evolve for a single clock cycle
                    self.state.apply(self.U1)

    def off_ramp_evolve(self, k: int):
        flipped_ramps = [self.flip_X_ramp(r) for r in self.ramp][::-1]
        
        for sequence in flipped_ramps:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.state.apply(self.U1)
                else:
                    # kick
                    self.state.apply(self.transmon_to_kick[k])

                    # free evolve for a single clock cycle
                    self.state.apply(self.U1)

    @staticmethod
    def flip_X_ramp(ramp: str):
        n = len(ramp)
        flipped = ['0'] * n
        for i in range(n):
            if ramp[i] == '1':
                new_pos = (n - i) % n
                flipped[new_pos] = '1'
        return ''.join(flipped)
    
    @staticmethod
    def truncate(operator: Operator, n_trunc: int):
        truncated_basis_to_matrix = {}
        
        for basis, matrix in operator.basis_to_matrix.items():
            truncated_basis_to_matrix[basis] = matrix[:n_trunc, :n_trunc]
        
        return Operator(truncated_basis_to_matrix)
    
    @staticmethod
    def upgrade(operator: Operator, n_trunc: int, idx: int, num_subsystems: int):
        upgraded_basis_to_matrix = {}
        
        for basis, matrix in operator.basis_to_matrix.items():
            upgraded_matrix = 1.0
            
            for k in range(num_subsystems): 
                if k == idx:
                    upgraded_matrix = np.kron(upgraded_matrix, matrix)
                else:
                    upgraded_matrix = np.kron(upgraded_matrix, np.eye(n_trunc))
                    
            upgraded_basis_to_matrix[basis] = upgraded_matrix
                
        return Operator(upgraded_basis_to_matrix)
    
    # NOTE: THIS IS WRONG 
    # @staticmethod
    # def create_projector(bases: list[str], n_trunc: int, idx: int, num_subsystems: int):
    #     projected_basis_to_matrix = {}
        
    #     for basis in bases:
    #         p = 1.0
            
    #         for k in range(num_subsystems): 
    #             if k == idx:
    #                 upgraded_matrix = np.kron(p, np.eye(n_trunc))
    #             else:
    #                 upgraded_matrix = np.kron(upgraded_matrix, np.zeros((n_trunc, n_trunc)))
                    
    #         projected_basis_to_matrix[basis] = p
                
    #     return Operator(projected_basis_to_matrix)
        