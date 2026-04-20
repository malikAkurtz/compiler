import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy

import numpy as np
from scipy.linalg import expm
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt

from quantum.System import System
from quantum.Operator import Operator
from quantum.Wavefunction import Wavefunction
from quantum.utils import *
from core.constants import *
from quantum.fidelity import *
from core.Circuit import Circuit
from core.Quantize import quantize
from core.Branch import *
from circuits.LCResonatorCircuit import LCResonatorCircuit
from quantum.Matrices import X, Y, Z


def main():
    # ---- Shared Hyper-parameters ----
    n                  = 7            # Number of charge states, -n/2 : n/2
    n_trunc            = 7              # number of states to truncate to 
    THETAS             = [0.03]         # U_kick angle
    clock_multiplier   = 8
    ramp               = []
    
    # ---- Hyper-parameters for Naive Harmonic Oscillator Qubit ----
    C          = 100e-15 # [F]
    L          = 10e-9   # [H]

    # ---- Create Ground Node ----
    gnd = Node(branches=[])
    
    lc_resonator = LCResonatorCircuit(
        gnd=gnd,
        capacitance=C,
        inductance=L
    )

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = lc_resonator.graph
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)    
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix = quantize(circuit=circuit, n=n)
    
    harmonic_oscillator = subsystems[0]
    
    n_full = n_trunc ** len(subsystems)
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_full - 1) * [0])})

    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : z["energy"].copy()})
    
    # ---- Add its its projection onto the flux basis
    initial_state["flux"] = vector_change_basis(
            harmonic_oscillator.flux_states,
            initial_state["energy"],
        )
    
    initial_state.U["flux"] = matrix_change_basis(
        harmonic_oscillator.flux_states,
        initial_state.U["energy"]
    )
    
    # ---- Creating The Full System ----
    system = System(
        subsystems=subsystems,
        dcsquids=[],
        EC_matrix=EC_matrix,
        thetas=THETAS,
        clock_multiplier=clock_multiplier,
        initial_state=initial_state,
        ramp=ramp,
        PHI_off=[], 
        PHI_on=[]
    )

    wavefuntions = []
    
    # ---- Want Every Fock State Projected in the Position Basis ----
    # 1) Store the position representation
    # 2) Apply creation operator to get next eigenstate of fock basis
    
    for n in range(n_full):
        
        wavefuntions.append(copy.deepcopy(system.state))

        if n < n_full - 1:
            system.state.apply(operator=harmonic_oscillator.creation)        
        
    # ---- Plot wavefunctions at their energy levels ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot potential energy
    V = 0.5 * C * harmonic_oscillator.angular_frequency**2 * harmonic_oscillator.fluxes**2
    ax.plot(harmonic_oscillator.fluxes / REDUCED_FLUX_QUANTUM, V, color='black', linewidth=2, label="V(x)")

    for n in range(n_full):
        energy = (hbar * harmonic_oscillator.angular_frequency) * (n + 0.5)
        probs = wavefuntions[n].get_probabilities(basis="flux")
        scaled = probs / np.max(probs) * (hbar * harmonic_oscillator.angular_frequency) * 0.35
        ax.fill_between(harmonic_oscillator.fluxes / REDUCED_FLUX_QUANTUM, energy, energy + scaled, alpha=0.3)
        ax.plot(harmonic_oscillator.fluxes / REDUCED_FLUX_QUANTUM, energy + scaled, linewidth=1.5, label=f"|{n}⟩")

    ax.set_xlabel("Flux", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    ax.set_title("Harmonic Oscillator Eigenstates in Position Basis", fontsize=16)
    ax.set_xlim(harmonic_oscillator.fluxes[0] / REDUCED_FLUX_QUANTUM, harmonic_oscillator.fluxes[-1] / REDUCED_FLUX_QUANTUM)
    ax.set_ylim(0, harmonic_oscillator.energies[-1])

    plt.tight_layout()
    plt.show()
        
    
if __name__=="__main__":
    main()