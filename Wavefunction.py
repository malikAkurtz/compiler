from __future__ import annotations
import numpy as np

class Wavefunction():
    """
    A wavefunction is a vector in a complex vector space,
    along with a specified basis
    """
    
    def __init__(self, probability_amplitudes: np.ndarray, basis: str):
        self.probability_amplitudes = probability_amplitudes
        self.basis                  = basis
        
    def apply(self, U: np.ndarray):
        new_probability_amplitudes = (U @ self.probability_amplitudes.reshape(-1, 1)).flatten()
        return Wavefunction(probability_amplitudes=new_probability_amplitudes, basis=self.basis)
        
    def get_probabilities(self):
        return np.array([np.abs(prob_amp)**2 for prob_amp in self.probability_amplitudes])
    
    def get_phase_projection(self, phases: np.ndarray):
        if self.basis == "fock":
            N = len(self.probability_amplitudes)
            K = len(phases)
            
            linear_map = np.zeros((K, N), dtype=complex)
            
            for k in range(K):
                phase = phases[k]
                for n in range(N):
                    linear_map[k][n] = np.exp(-1j * n * phase) / np.sqrt(2*np.pi)
                    
            return Wavefunction(probability_amplitudes=(linear_map @ self.probability_amplitudes),basis="phase")