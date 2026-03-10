import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


from constants import *
from HarmonicOscillator import HarmonicOscillator
from Wavefunction import Wavefunction
from utils import *

def main():
    # ---- Hyper-parameters for Harmonic Oscillator ----
    capacitance     = 100     # [fF]
    inductance      = 10      # [nH]
    n_cut = 5      # Number of Fock states to truncate to
    
    # ---- Derived Physical Constants ----
    spring_constant   = 1 / inductance
    mass              = capacitance
    angular_frequency = np.sqrt(spring_constant / mass) # [rad/s]
    
    harmonic_oscillator = HarmonicOscillator(
                            mass=mass, 
                            angular_frequency=angular_frequency, 
                            n_cut=n_cut
                            )
    
    print(f"Creation and Annihilation Operators in Fock/Energy Basis: ")
    print(f"Annihilation: ")
    print(harmonic_oscillator.annihilation)
    print(f"Creation: ")
    print(harmonic_oscillator.creation)
    print(f"Number Operator in Fock/Energy Basis: ")
    print(harmonic_oscillator.n)
    print("Hamiltonian in Fock/Energy Basis: ")
    print(harmonic_oscillator.H)
    print("Hamiltonian Energies: ")
    print(harmonic_oscillator.energies)
    
    # ---- Creating Our Initial Quantum State ----
    probability_amplitudes    = n_cut * [0]
    probability_amplitudes[0] = 1
    probability_amplitudes    = np.array(probability_amplitudes)
    probability_amplitudes    = probability_amplitudes / np.linalg.norm(probability_amplitudes)
    
    current_state_fock = Wavefunction(probability_amplitudes=probability_amplitudes, basis="fock")
    
    # eigenvectors of number operator n projected onto the position basis
    eigenstates = []
    energies    = []
    
    for n in range(n_cut):
        current_state_pos = current_state_fock.change_of_basis(
            transformation_matrix=harmonic_oscillator.position_states.conj().T,
            basis="position"
            )
        eigenstates.append(current_state_pos)
        energies.append((n + 0.5))
        
        next_state_fock = current_state_fock.apply(U=harmonic_oscillator.creation.matrix)
        next_state_fock.probability_amplitudes = next_state_fock.probability_amplitudes / np.linalg.norm(next_state_fock.probability_amplitudes)
        
        
        current_state_fock = next_state_fock
        
        
    # ---- Plot wavefunctions at their energy levels ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot potential energy
    V = 0.5 * mass * angular_frequency**2 * harmonic_oscillator.positions**2 / (hbar * angular_frequency)
    ax.plot(harmonic_oscillator.positions, V, color='black', linewidth=2, label="V(x) = ½mω²x²")

    for n in range(n_cut):
        energy = n + 0.5  # dimensionless energy levels
        probs = np.abs(eigenstates[n].probability_amplitudes)**2
        scaled = probs / np.max(probs) * 0.35
        ax.fill_between(harmonic_oscillator.positions, energy, energy + scaled, alpha=0.3)
        ax.plot(harmonic_oscillator.positions, energy + scaled, linewidth=1.5, label=f"|{n}⟩")
        ax.axhline(y=energy, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel("Position", fontsize=14)
    ax.set_ylabel("Energy (ℏω)", fontsize=14)
    ax.set_title("Harmonic Oscillator Eigenstates in Position Basis", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(harmonic_oscillator.positions[0], harmonic_oscillator.positions[-1])
    ax.set_ylim(0, n_cut)

    plt.tight_layout()
    plt.show()
        
    
if __name__=="__main__":
    main()