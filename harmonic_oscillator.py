import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


from constants import *
from HarmonicOscillator import HarmonicOscillator
from Wavefunction import Wavefunction
from utils import *

def main():
    # ---- Hyper-parameters for Harmonic Oscillator ----
    capacitance     = 100e-15 # [F]
    inductance      = 10e-9   # [H]
    n_cut = 20      # Number of Fock states to truncate to
    
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
    
    current_state = Wavefunction(basis_to_coefs={"fock": probability_amplitudes})
    
    # eigenvectors of number operator n (Fock states) projected onto the position basis
    eigenstates = []
    
    for n in range(n_cut):
        pos_coefs = vector_change_basis(
            transformation_matrix=harmonic_oscillator.position_states,
            vector=current_state.get_projection(basis="fock")
            )
        current_state.set_projection(basis="position", coefs=pos_coefs)
        
        eigenstates.append(current_state)

        if n < n_cut - 1:
            next_state = current_state.apply(operator=harmonic_oscillator.creation)
            for basis, coefs in next_state.basis_to_coefs.items():
                next_state.set_projection(basis=basis, coefs=(coefs / np.linalg.norm(coefs)))

            current_state = next_state
        
        
    # ---- Plot wavefunctions at their energy levels ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot potential energy
    V = 0.5 * mass * angular_frequency**2 * harmonic_oscillator.positions**2
    ax.plot(harmonic_oscillator.positions / PHI_0, V, color='black', linewidth=2, label="V(x) = ½mω²x²")

    for n in range(n_cut):
        energy = (hbar * angular_frequency) * (n + 0.5)
        probs = eigenstates[n].get_probabilities(basis="position")
        scaled = probs / np.max(probs) * (hbar * angular_frequency) * 0.35
        ax.fill_between(harmonic_oscillator.positions / PHI_0, energy, energy + scaled, alpha=0.3)
        ax.plot(harmonic_oscillator.positions / PHI_0, energy + scaled, linewidth=1.5, label=f"|{n}⟩")

    ax.set_xlabel("Position", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    ax.set_title("Harmonic Oscillator Eigenstates in Position Basis", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(harmonic_oscillator.positions[0] / PHI_0, harmonic_oscillator.positions[-1] / PHI_0)
    ax.set_ylim(harmonic_oscillator.energies[0], harmonic_oscillator.energies[-1])

    plt.tight_layout()
    plt.show()
        
    
if __name__=="__main__":
    main()