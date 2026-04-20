import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy

import numpy as np
from scipy.linalg import expm
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt

from quantum.System import System
from quantum.Wavefunction import Wavefunction
from quantum.utils import *
from core.constants import *
from quantum.fidelity import *
from core.Circuit import Circuit
from core.Quantize import quantize
from core.Branch import *
from quantum.Matrices import X, Y, Z
from circuits.DCSQUIDCircuit import DCSQUIDCircuit
from circuits.TransmonCircuit import TransmonCircuit


def main():
    # ---- Shared Hyper-parameters ----
    n                  = 201            # Number of charge states, -n/2 : n/2 for each transmon
    n_trunc            = 7              # number of states to truncate to for each transmon
    THETAS             = np.array([0.03])
    clock_multiplier   = 8
    ramp               = ['01000000', '11000000', '10000000', '00000000', '00000000']
    # ramp               = []
    
    # ---- Transmon Circuit Hyper-parameters ----
    EJ_EC = 69
    EC    = h * 250 * 1e6   # Charging energy [J]
    
    # ---- Derived Physical Constants ----
    EJ    = EJ_EC * EC
    JL    = EJ / (2 * REDUCED_FLUX_QUANTUM)
    JR    = EJ / (2 * REDUCED_FLUX_QUANTUM)
    n_zpf = (1/2) * (EJ_EC / 2)**(1/4) # approximation
    BETAS = THETAS / (2 * n_zpf)        # approximation
    C     = e**2 / (2 * EC)            # includes Josephson Capacitance CJ
    C_C   = (BETAS[0] * hbar * C) / FLUX_QUANTUM
    C_S   = C - C_C
    
    # ---- Create Ground Node ----
    gnd = Node(branches=[])
    
    # ---- Create Nodes and Branches of Each DCSQUID Circuit (C_JL, C_JR are embedded in C_S) ----
    q1_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=0,
        left_josephson_current=JL,
        right_josephson_current=JR,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    # ---- Create Transmon Circuit by Adding a Shunt Capacitor Branch ----
    q1 = TransmonCircuit(
        gnd=gnd,
        dcsquid=q1_dcsquid,
        shunt_capacitance=C_S,
        coupling_capacitance=C_C
    )

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = Graph(vertices=[gnd, q1.island], edges=q1.branches)
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix = quantize(circuit=circuit, n=n)
    
    transmon = subsystems[0]
        
    n_full = n_trunc ** len(subsystems)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    n_plot = 7

    wavefunctions = []
    state = np.zeros(n)
    state[0] = 1.0

    for k in range(n_plot):
        flux_coefs = vector_change_basis(transmon.flux_states, state)
        wavefunctions.append(np.abs(flux_coefs)**2)
        
        if k < n_plot - 1:
            state = transmon.creation["fock"] @ state
            state /= np.linalg.norm(state)

    fig, ax = plt.subplots(figsize=(10, 8))

    V = EJ * (1 - np.cos(transmon.fluxes / REDUCED_FLUX_QUANTUM))
    E_shift = transmon.energies[0] + EJ

    ax.plot(transmon.fluxes / REDUCED_FLUX_QUANTUM, V, color='black', linewidth=2)

    for k in range(n_plot):
        energy = transmon.energies[k] - transmon.energies[0] + E_shift
        probs = wavefunctions[k]
        scaled = probs / np.max(probs) * (transmon.energies[1] - transmon.energies[0]) * 0.35
        ax.fill_between(transmon.fluxes / REDUCED_FLUX_QUANTUM, energy, energy + scaled, alpha=0.3)
        ax.plot(transmon.fluxes / REDUCED_FLUX_QUANTUM, energy + scaled, linewidth=1.5, label=f"|{k}⟩")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, transmon.energies[n_plot-1] - transmon.energies[0] + E_shift + EJ*0.1)
    ax.set_xlabel("φ", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    ax.set_title("Transmon Eigenstates", fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()