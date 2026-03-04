import numpy as np

from Basis import Basis

class Wavefunction():
    """
    A wavefunction is a vector in a complex vector space,
    along with a specified basis
    """
    
    def __init__(self, probability_amplitudes: np.ndarray, basis: Basis):
        self.probability_amplitudes = probability_amplitudes
        self.basis                  = basis
        
    def apply(self, U: np.ndarray):
        self.probability_amplitudes = (U @ self.probability_amplitudes.reshape(-1, 1)).flatten()
        
    def get_probabilities(self):
        return np.array([np.abs(prob_amp)**2 for prob_amp in self.probability_amplitudes])