import numpy as np
from scipy.linalg import expm

from Transmon import Transmon
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, EC: float, EJ_EC: float, n_cut: int, theta: float, initial_state: Wavefunction, dim_sub: int):
        self.n_cut = n_cut
        self.dim_sub = dim_sub
        self.transmon = Transmon(EC=EC,
                                 EJ_EC=EJ_EC,
                                 n_cut=n_cut,
                                 dim_sub=dim_sub
                                 )
        
        self.C = (2 * EC) / (e**2)
        self.CC = theta / ( PHI_0 * np.sqrt( (2*self.transmon.qubit_angular_frequency) / self.C ) )
        
        self.sfq_driver = SFQDriver(CC=self.CC,
                                    omega_q=(self.transmon.qubit_frequency * (2*np.pi)),
                                    C=((2 * EC) / (e**2)),
                                    EJ=(EJ_EC * EC),
                                    EC=EC,
                                    n=self.transmon.n,
                                    )
        
        self.state = initial_state
        
        self.T = self.T = (2*np.pi) / self.transmon.qubit_angular_frequency # Qubit period [s]
        # self.U_free_1 = expm((-1j * (self.transmon.H0.matrix - self.transmon.omega_q * (self.transmon.a_dagger.matrix @ self.transmon.a.matrix)) * (1*(self.T/4))) / hbar)
        # self.U_free_1 = expm((-1j * (self.transmon.H0.matrix - self.transmon.omega_q * (self.transmon.a_dagger.matrix @ self.transmon.a.matrix)) * (2*(self.T/4))) / hbar)
        # self.U_free_1 = expm((-1j * (self.transmon.H0.matrix - self.transmon.omega_q * (self.transmon.a_dagger.matrix @ self.transmon.a.matrix)) * (3*(self.T/4))) / hbar)
        # self.U_free_1 = expm((-1j * (self.transmon.H0.matrix - self.transmon.omega_q * (self.transmon.a_dagger.matrix @ self.transmon.a.matrix)) * (4*(self.T/4))) / hbar)
        
    def free_evolve(self, duration: int):
        if duration == 1:
            self.state = self.state.apply(U=self.U_free_1)
        elif duration == 2:
            self.state = self.state.apply(U=self.U_free_2)
        elif duration == 3:
            self.state = self.state.apply(U=self.U_free_3)
        elif duration == 4:
            self.state = self.state.apply(U=self.U_free_4)

    def RY(self, theta_target: float):
        U_target = np.array([
            [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
            [np.sin(theta_target / 2), np.cos(theta_target / 2)]
        ])
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = np.eye(N=self.dim_sub)
        for _ in range(N):
            self.state = self.sfq_driver.apply_pulse(self.state)
            U = self.sfq_driver.U_kick @ U
            
            self.free_evolve(duration=4)
            U = self.U_free_4 @ U
        
        return U, U_target
    
    def RX(self, theta_target: float):        
        U_target = np.array([
            [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
            [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
        ])
        
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        U = np.eye(N=self.dim_sub)
        for _ in range(N):
            self.free_evolve(duration=3)
            U = self.U_free_3 @ U
            
            self.state = self.sfq_driver.apply_pulse(self.state)
            U = self.sfq_driver.U_kick @ U
            
            self.free_evolve(duration=1)
            U = self.U_free_1 @ U
            
        return U, U_target
    
    def X(self):
        return self.RX(theta_target=np.pi) # ONLY FOR SINGLE QUBIT CASE
            
    def Hadamard(self):
        U_target = np.array([
            [1, 1],
            [1, -1]
        ])
        U_target *= 1 / np.sqrt(2)
        
        U = np.eye(N=self.n_cut)
        # RY(pi/2) rotation
        RY, _ = self.RY(np.pi/2)
        U = RY @ U
        # RZ(pi) rotation
        self.free_evolve(duration=2)
        U = self.U_free_2 @ U
        
        return U, U_target

        