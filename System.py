import numpy as np
from scipy.linalg import expm

from Operator import Operator
from Transmon import Transmon
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, EC: float, EJ_EC: float, n_cut: int, theta: float, initial_state: Wavefunction, fock_approximation: bool):
        self.fock_approx = fock_approximation
        self.n_cut = n_cut
        self.transmon = Transmon(EC=EC,
                                 EJ_EC=EJ_EC,
                                 n_cut=n_cut,
                                 fock_approximation=fock_approximation
                                 )
        
        self.C = (2 * EC) / (e**2)
        self.CC = theta / ( PHI_0 * np.sqrt( (2*self.transmon.qubit_angular_frequency) / self.C ) )
        
        self.sfq_driver = SFQDriver(CC=self.CC,
                                    omega_q=(self.transmon.qubit_frequency * (2*np.pi)),
                                    C=((2 * EC) / (e**2)),
                                    EJ=(EJ_EC * EC),
                                    EC=EC,
                                    n=self.transmon.n,
                                    fock_approximation=fock_approximation
                                    )
        
        self.state = initial_state
        
        self.T = (2*np.pi) / self.transmon.qubit_angular_frequency # Qubit period [s]
        
        # H_rot = self.transmon.H0.get_projection("energy") \
        #         - hbar * self.transmon.qubit_angular_frequency * self.transmon.N.get_projection("energy")
        
        if fock_approximation:
            self.U_free_1 = Operator({"energy" if not self.fock_approx else "fock": expm(-1j * self.transmon.H0.get_projection("fock") * (1 * self.T / 4) / hbar)})
            self.U_free_2 = Operator({"energy" if not self.fock_approx else "fock": expm(-1j * self.transmon.H0.get_projection("fock") * (2 * self.T / 4) / hbar)})
            self.U_free_3 = Operator({"energy" if not self.fock_approx else "fock": expm(-1j * self.transmon.H0.get_projection("fock") * (3 * self.T / 4) / hbar)})
            self.U_free_4 = Operator({"energy" if not self.fock_approx else "fock": expm(-1j * self.transmon.H0.get_projection("fock") * (4 * self.T / 4) / hbar)})
        
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
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = Operator({"energy" if not self.fock_approx else "fock": np.eye(N=self.n_cut)})
        for _ in range(N):
            self.state = self.sfq_driver.apply_pulse(self.state)
            U.set_projection(
                basis="energy" if not self.fock_approx else "fock",
                matrix=self.sfq_driver.U_kick.get_projection("energy" if not self.fock_approx else "fock") \
                    @ U.get_projection("energy" if not self.fock_approx else "fock")
            )
            
            self.free_evolve(duration=4)
            U.set_projection(
                basis="energy" if not self.fock_approx else "fock",
                matrix=self.U_free_4.get_projection("energy" if not self.fock_approx else "fock") \
                    @ U.get_projection("energy" if not self.fock_approx else "fock")
            )
        
        return U, U_target
    
    def RX(self, theta_target: float):        
        U_target = np.array([
            [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
            [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
        ])
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = Operator({"energy" if not self.fock_approx else "fock" : np.eye(N=self.n_cut)})
        for _ in range(N):
            self.free_evolve(duration=3)
            U.set_projection(
                basis="energy" if not self.fock_approx else "fock",
                matrix=self.U_free_3.get_projection("energy" if not self.fock_approx else "fock") \
                    @ U.get_projection("energy" if not self.fock_approx else "fock")
            )
            
            self.state = self.sfq_driver.apply_pulse(self.state)
            U.set_projection(
                basis="energy" if not self.fock_approx else "fock",
                matrix=self.sfq_driver.U_kick.get_projection("energy" if not self.fock_approx else "fock") \
                    @ U.get_projection("energy" if not self.fock_approx else "fock")
            )
            
            self.free_evolve(duration=1)
            U.set_projection(
                basis="energy" if not self.fock_approx else "fock",
                matrix=self.U_free_1.get_projection("energy" if not self.fock_approx else "fock") \
                    @ U.get_projection("energy" if not self.fock_approx else "fock")
            )
            
        return U, U_target
    
    def X(self):
        return self.RX(theta_target=np.pi) # ONLY FOR SINGLE QUBIT CASE
            
    def Hadamard(self):
        U_target = np.array([
            [1, 1],
            [1, -1]
        ])
        U_target *= 1 / np.sqrt(2)
        
        U = Operator({"energy" if not self.fock_approx else "fock" : np.eye(N=self.n_cut)})
        # RY(pi/2) rotation
        RY, _ = self.RY(np.pi/2)
        U = RY.get_projection("energy" if not self.fock_approx else "fock") @ U.get_projection("energy" if not self.fock_approx else "fock")
        # RZ(pi) rotation
        self.free_evolve(duration=2)
        U = self.U_free_2.get_projection("energy" if not self.fock_approx else "fock") @ U.get_projection("energy" if not self.fock_approx else "fock")
        
        return U, U_target

        