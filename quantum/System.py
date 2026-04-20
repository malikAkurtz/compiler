from __future__ import annotations
import numpy as np
from scipy.linalg import expm
import copy

from quantum.Transmon import Transmon
from quantum.Operator import Operator
from quantum.Wavefunction import Wavefunction
from core.constants import *
from circuits.DCSQUIDCircuit import DCSQUIDCircuit
from quantum.QuantumOscillator import QuantumOscillator
from quantum.LCResonator import HarmonicOscillator

class System():
    def __init__(self, subsystems: list[QuantumOscillator], dcsquids: list[DCSQUIDCircuit], EC_matrix: np.ndarray, thetas: np.ndarray, clock_multiplier: int, initial_state: Wavefunction, ramp: list[str], PHI_off: np.ndarray, PHI_on: np.ndarray):
        self.subsystems    = subsystems
        self.dcsquids      = dcsquids
        self.EC_matrix     = EC_matrix
        self.thetas        = thetas
        self.M             = clock_multiplier
        self.state         = initial_state  
        self.ramp          = ramp
        self.PHI_off       = PHI_off
        self.PHI_on        = PHI_on
        
        self.n_charge      = len(self.subsystems[0].n["energy"])
        self.n_full        = len(self.state["energy"]) # Dimension of full Hilbert space of the system
        self.n_trunc       = int(round(self.n_full**(1/len(self.subsystems))))
        
        # ---- Derive Qubit and Clock Periods ----
        self.T_q           = np.array([(2*np.pi) / s.angular_frequency for s in self.subsystems]) # Qubit period [s]
        # We assume our computational qubits share the same angular frequency s.t. the clock period is well-defined
        self.T_c           = self.T_q[0] / self.M                                                      # Clock period [s]
        
        # ---- Create New Subsystems Specifically For the System (Embedded in Larger Hilbert Space of Dimension n_full)
        self.sys_subsystems = copy.deepcopy(self.subsystems)
        
        for k in range(len(self.subsystems)):
            sys_subsystem = self.sys_subsystems[k]
            
            for op_label in list(sys_subsystem.operators.keys()):
                
                sys_subsystem.operators[op_label] = System.upgrade(
                    operator=System.truncate(sys_subsystem.operators[op_label], self.n_trunc),
                    n_trunc=self.n_trunc,
                    idx=k,
                    num_subsystems=len(self.subsystems)
                )

        # ---- Derive Kick Operators for Logical Qubits
        self.transmon_to_kick = {}
        
        for k in range(len(self.subsystems)):
            
            if isinstance(self.subsystems[k], Transmon):
                self.transmon_to_kick[k] = Operator(
                            basis_to_matrix={"energy": expm(-1j * (thetas[k] / self.subsystems[k].r) / 2 * self.subsystems[k].n["energy"])}
                        )
            elif isinstance(self.subsystems[k], HarmonicOscillator):
                self.transmon_to_kick[k] = Operator(
                            basis_to_matrix={"energy": expm((self.thetas[k] / 2) * (self.subsystems[k].creation["energy"] - self.subsystems[k].annihilation["energy"]))}
                        )
            
            self.transmon_to_kick[k] = System.truncate(
                operator=self.transmon_to_kick[k],
                n_trunc=self.n_trunc
            )
            
            self.transmon_to_kick[k] = System.upgrade(
                operator=self.transmon_to_kick[k],
                n_trunc=self.n_trunc,
                idx=k,
                num_subsystems=len(self.subsystems)
            )
        
        # ---- Derive Coupling Hamiltonain HC ----
        coupling_sum = 0
        for k in range(len(self.subsystems)):
            for l in range(len(self.subsystems)):
                if k == l:
                    continue
                else:   
                    coupling_sum += (self.EC_matrix[k][l] * self.sys_subsystems[k].operators["n"]["energy"] @ self.sys_subsystems[l].operators["n"]["energy"])
                
        self.HC = Operator(
            basis_to_matrix={"energy" : 4 * coupling_sum}
        )
        
        # ---- Derive the Unperturbed Hamiltonian H0 ----
        total_H0 = sum(self.sys_subsystems[j].operators["H0"]["energy"] for j in range(len(self.subsystems)))
        self.H0 = Operator(
            basis_to_matrix={"energy": total_H0 + self.HC["energy"]}
        )
        
        # ---- For single qubit case ----
        self.U_1 = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (1 * self.T_c) / hbar)}
        )
        
        self.U_M = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (self.M * self.T_c) / hbar)}
        )
                        
    def set_coupler_flux(self, k: int, EJ_new: float, n: int):
        # ---- Create a new transmon with an updated josephson energy ----
        new_transmon = Transmon(
            EC=self.subsystems[k].EC,
            EJ_EC=(EJ_new/self.subsystems[k].EC),
            n=n
        )
        
        # ---- Update sys_subsystems[k]'s operators with new_transmon's operators ----
        for op_label, operator in new_transmon.operators.items():
            self.sys_subsystems[k].operators[op_label] = System.upgrade(
                operator=System.truncate(operator, self.n_trunc),
                n_trunc=self.n_trunc,
                idx=k,
                num_subsystems=len(self.subsystems)
            )
            
        # ---- Rebuild Coupling Hamiltonain HC ----
        coupling_sum = 0
        for i in range(len(self.subsystems)):
            for j in range(len(self.subsystems)):
                if i == j:
                    continue
                else:   
                    coupling_sum += (self.EC_matrix[i][j] * self.sys_subsystems[i].operators["n"]["energy"] @ self.sys_subsystems[j].operators["n"]["energy"])
                
        self.HC = Operator(
            basis_to_matrix={"energy" : 4 * coupling_sum}
        )
        
        # ---- Rebuild the Unperturbed Hamiltonian H0 ----
        total_H0 = sum(self.sys_subsystems[j].operators["H0"]["energy"] for j in range(len(self.subsystems)))
        self.H0 = Operator(
            basis_to_matrix={"energy": total_H0 + self.HC["energy"]}
        )
        
        # ---- Rebuild free evolution unitaries ----
        self.U_1 = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (1 * self.T_c) / hbar)}
        )
        
        self.U_M = Operator(
            basis_to_matrix={"energy": expm(-1j * self.H0["energy"] * (self.M * self.T_c) / hbar)}
        )            
        
        
    def free_evolve(self, clock_cycles: int):
        self.state.apply(
            operator=Operator(
                {"energy": expm(-1j * self.H0["energy"] * (clock_cycles * self.T_c) / hbar)}
                )
            )

    def RY(self, k: int, theta_target: float):
        
        self.on_ramp_evolve(k)
        
        N_kicks = int(np.round(theta_target/self.thetas[k]))
        # N_kicks = 47
                
        for _ in range(N_kicks):
            self.state.apply(self.transmon_to_kick[k])
            
            self.state.apply(self.U_M)
        
        self.off_ramp_evolve(k)
        
    def RX(self, k: int, theta_target: float):
        
        # RZ(pi/2)
        for _ in range(int((1 * self.M) / 4)):
            self.state.apply(self.U_1)
        
        # RY(theta_target)
        self.RY(k, theta_target)
        
        # RZ(-pi/2) = RZ(3pi/2)
        for _ in range(int((3 * self.M) / 4)):
            self.state.apply(self.U_1)
            
    def fSim(self, duration):
        self.set_coupler_flux(
            k=1, 
            EJ_new=DCSQUIDCircuit.calculate_effective_EJ(
                PHI_ext=self.PHI_on[1], 
                JL=self.dcsquids[1].J_L, 
                JR=self.dcsquids[1].J_R
                ),
            n=self.n_charge
            )
        
        for _ in range(int(round(duration / self.T_c))):
            self.state.apply(self.U_1)
            
        self.set_coupler_flux(
            k=1, 
            EJ_new=DCSQUIDCircuit.calculate_effective_EJ(
                PHI_ext=self.PHI_off[1], 
                JL=self.dcsquids[1].J_L, 
                JR=self.dcsquids[1].J_R
                ),
            n=self.n_charge
            )
    
    def CZ(self):
        # Correspond to hold duration of 17 ns
        theta = 0.7732102591991971
        phi = -3.1294047493918216
        
        alpha = np.arcsin(np.sqrt(((1/2) - np.sin(phi/2)**2) / (np.sin(theta)**2 - np.sin(phi/2)**2)))
        eta   = np.arctan((np.tan(alpha)*np.sin(theta)) / (np.sin(phi/2))) + ((np.pi/2) * (1 - np.sign(np.sin(phi/2))))
        eps   = np.arctan((np.tan(alpha)*np.cos(theta)) / (np.cos(phi/2))) + ((np.pi/2) * (1 - np.sign(np.cos(phi/2))))
        
        self.RX(k=0, theta_target=eps)
        self.RX(k=2, theta_target=eta)
        
        self.fSim(duration=17e-9)
        
        self.RX(k=0, theta_target=2*alpha)
        
        
            
    def on_ramp_evolve(self, k: int):
    
        for sequence in self.ramp:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.state.apply(self.U_1)
                else:
                    # kick
                    self.state.apply(self.transmon_to_kick[k])

                    # free evolve for a single clock cycle
                    self.state.apply(self.U_1)

    def off_ramp_evolve(self, k: int):
        flipped_ramps = [self.flip_X_ramp(r) for r in self.ramp][::-1]
        
        for sequence in flipped_ramps:
            for action in sequence:
                if action == "0":
                    # free evolve for a single clock cycle
                    self.state.apply(self.U_1)
                else:
                    # kick
                    self.state.apply(self.transmon_to_kick[k])

                    # free evolve for a single clock cycle
                    self.state.apply(self.U_1)

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