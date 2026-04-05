###############################################################################
#
#   quantization.py
#
#   Builds the quantum Hamiltonian in the charge or Fock basis, diagonalizes
#   it, and transforms operators into the energy eigenbasis.
#
###############################################################################


import numpy as np
from constants import *
from Circuit import Circuit
from Operator import Operator
from QuantumOscillator import QuantumOscillator
from utils import create_upper_lower

def quantize(circuit: Circuit, n_cut: int):
    """
    1) Builds charge operator                     (charge basis + Fock basis)
    2) Builds creation and annihilation operators (Fock basis)
    3) Builds kinetic energy operator             (charge basis)
    4) Builds potential energy operator           (charge basis)
    5) Builds unperturbed Hamiltonian operator    (charge basis + Fock basis)
    6) Diagonalizes unperturbed Hamiltonian       (charge basis)
    7) Calculate anharmonicity, qubit frequency, 
        and qubit angular frequency

    Parameters
    ----------
    circuit : Circuit
        A fully constructed Circuit object.
    n_cut : int
        Hilbert-space truncation: Cooper-pair numbers run from -n_cut to +n_cut
        (charge basis), or Fock states 0..n_cut-1 (Fock basis).

    Returns
    -------
    QuantizationResult
    """
    if circuit.N != 1:
        raise NotImplementedError("Quantization currently supports single-node circuits only")

    C_T   = circuit.capacitance_matrix[0][0]
    EC    = e**2 / (2 * C_T)
    EJ    = circuit.josephson_elements[0].EJ
    EJ_EC = EJ / EC

    # Charge operator
    n = Operator(
        basis_to_matrix={"charge" : np.diag([n for n in range(int(-(n_cut-1)/2), int((n_cut-1)/2) + 1)])}
    )   
    
    annihilation, creation  = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
    
    n_zpf = (1/2)*(EJ_EC / 2)**(1/4) # charge zero-point-fluctuation
    
    # Used in kick operator. This is an approximation
    n["fock"] = 1j * n_zpf * (creation["fock"] - annihilation["fock"])
    
    # Kinetic energy operator
    T = Operator(
        basis_to_matrix={"charge" : 4*EC*(n["charge"]**2)}
    ) 
    
    # Potential energy operator
    V = Operator(
        basis_to_matrix={"charge" : create_upper_lower(value=-EJ / 2, dim=n_cut)}
    ) 
    
    # Unperturbed Hamiltonian operator
    H0 = Operator(
        basis_to_matrix={"charge" : T["charge"] + V["charge"]}
        ) 
    
    # Diagonalize the Hamiltonian in the charge basis to get energy eigenvalues/eigenvectors
    energies, energy_states = np.linalg.eigh(H0["charge"])
    
    # Unperturbed Hamiltonian operator in the energy basis
    H0["energy"] = np.diag(energies)

    anharmonicity           = (energies[2] - energies[1]) - (energies[1] - energies[0])
    qubit_frequency         = (energies[1] - energies[0]) / h 
    qubit_angular_frequency = qubit_frequency * (2*np.pi)
    
    # Used in Fock approximation. This is an approximation
    H0["fock"] = hbar * qubit_angular_frequency * \
                ( (creation["fock"] @ annihilation["fock"]) ) \
                - ( (anharmonicity / 2) * (creation["fock"] @ creation["fock"]) \
                    @ (annihilation["fock"] @ annihilation["fock"]))

    return n, n_zpf, creation, annihilation, H0, energies, energy_states, anharmonicity, qubit_frequency, qubit_angular_frequency
