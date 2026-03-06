import numpy as np
from scipy.linalg import expm

from Transmon import Transmon
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from constants import *

class System():
    def __init__(self, EC: float, EJ_EC: float, n_cut: int, theta: float, initial_state: Wavefunction):
        self.transmon = Transmon(EC=EC,
                                 EJ_EC=EJ_EC,
                                 n_cut=n_cut
                                 )
        
        self.C = (2 * EC) / (e**2)
        self.CC = theta / ( PHI_0 * np.sqrt( (2*self.transmon.omega_q) / self.C ) )
        
        self.sfq_driver = SFQDriver(CC=self.CC,
                                    omega_q=(self.transmon.fq * (2*np.pi)),
                                    C=((2 * EC) / (e**2)),
                                    H0=self.transmon.H0,
                                    a=self.transmon.a,
                                    a_dagger=self.transmon.a_dagger
                                    )
        
        self.state = initial_state
        
        self.T = self.T = (2*np.pi) / self.transmon.omega_q # Qubit period [s]
        self.U_free_1 = expm((-1j * self.transmon.H0.matrix * (1*(self.T/4))) / hbar)
        self.U_free_2 = expm((-1j * self.transmon.H0.matrix * (2*(self.T/4))) / hbar)
        self.U_free_3 = expm((-1j * self.transmon.H0.matrix * (3*(self.T/4))) / hbar)
        self.U_free_4 = expm((-1j * self.transmon.H0.matrix * (4*(self.T/4))) / hbar)
        
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
        N = int(np.round(theta_target / self.sfq_driver.theta)) + 1 # number of total kicks/sfq pulses
        
        for _ in range(N):
            self.state = self.sfq_driver.apply_pulse(self.state)
            self.free_evolve(duration=4)
            
    def Hadamard(self):        
        # RY(pi/2) rotation
        self.RY(np.pi/2)
        # RZ(pi) rotation
        self.free_evolve(duration=2)
        