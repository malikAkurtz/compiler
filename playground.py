import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from HarmonicOscillator import HarmonicOscillator
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *

def main():
    # ---- Hyper-parameters for Naive Harmonic Oscillator Qubit ----
    C          = 100e-15 # [F]
    L          = 10e-9   # [H]
    n_cut      = 41      # Number of charge states, -n_cut : n_cut
    theta      = 0.03    # U_kick angle
    # if basis = "fock", everything will be done in the fock basis
    # if basis = "energy" everything will be done in the energy basis (no fock approximation)
    # i.e. has to be "fock" for Harmonic Oscillator, but acts as a hyperparameter for a Transmon
    basis      = "fock" 
    
    # ---- Derived Physical Constants ----
    spring_constant   = 1 / L
    angular_frequency = np.sqrt(spring_constant / C) # [rad/s]
        
    # ---- Create the Oscillator and Driver ----
    oscillator = HarmonicOscillator(
        mass=C,
        angular_frequency=angular_frequency,
        n_cut=n_cut
    )
    
    print(oscillator.n[basis].shape)
    
    print(oscillator.N[basis].shape)
    
    print(np.allclose(oscillator.n[basis], oscillator.N[basis]))
    
if __name__=="__main__":
    main()