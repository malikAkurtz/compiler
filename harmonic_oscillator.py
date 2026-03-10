import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


from constants import hbar
from HarmonicOscillator import HarmonicOscillator
from Wavefunction import Wavefunction
from utils import *

def main():
    # ---- Hyper-parameters for Harmonic Oscillator ----
    C     = 100     # [fF]
    L     = 10      # [nH]
    n_cut = 20      # Number of Fock states to truncate to
    
    # ---- Derived Physical Constants ----
    k = 1 / L
    m = C
    angular_frequency = np.sqrt(k / m) # [rad/s]
    
    harmonic_oscillator = HarmonicOscillator(
                            mass=m, 
                            angular_frequency=angular_frequency, 
                            n_cut=n_cut
                            )
    
    print(f"Creation and Annihilation Operators in Fock/Energy basis: ")
    print(f"a: ")
    print(harmonic_oscillator.a)
    print(f"a_dagger: ")
    print(harmonic_oscillator.a_dagger)
    print(f"Number Operator n in Fock/Energy basis: ")
    print(harmonic_oscillator.n)
    print("Hamiltonian in Fock/Energy basis: ")
    print(harmonic_oscillator.H)
    
    # ---- Creating Our Initial Quantum State ----
    probability_amplitudes = n_cut * [0]
    probability_amplitudes[0] = 1
    probability_amplitudes = np.array(probability_amplitudes)
    probability_amplitudes = probability_amplitudes / np.linalg.norm(probability_amplitudes)
    
    current_state = Wavefunction(probability_amplitudes=probability_amplitudes, basis="fock")
    
    eigenstates = []
    energies    = []
    phases = np.linspace(-np.pi, np.pi, 100)
    
    for n in range(5):
        current_state_phase = current_state.get_phase_projection(phases=phases)
        eigenstates.append(current_state_phase)
        energies.append((n + 0.5))
        
        next_state = current_state.apply(U=harmonic_oscillator.a_dagger.matrix)
        next_state.probability_amplitudes = next_state.probability_amplitudes / np.linalg.norm(next_state.probability_amplitudes)
        
        
        current_state = next_state
        
        
    # ---- Plot wavefunctions at their energy levels ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot potential energy
    phi_potential = np.linspace(-np.pi, np.pi, 200)
    V = 0.5 * m * angular_frequency**2 * phi_potential**2
    ax.plot(phi_potential, V, color='black', linewidth=2, label="V(φ) = ½mω²φ²")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for n in range(5):
        energy = n + 0.5  # dimensionless energy levels
        probs = np.abs(eigenstates[n].probability_amplitudes)**2
        scaled = probs / np.max(probs) * 0.35
        ax.fill_between(phases, energy, energy + scaled, alpha=0.3, color=colors[n])
        ax.plot(phases, energy + scaled, color=colors[n], linewidth=1.5, label=f"|{n}⟩")
        ax.axhline(y=energy, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel("Phase φ", fontsize=14)
    ax.set_ylabel("Energy (ℏω)", fontsize=14)
    ax.set_title("Harmonic Oscillator Eigenstates in Phase Basis", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.show()
        
    
if __name__=="__main__":
    main()