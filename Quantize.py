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
    annihilation, creation  = QuantumOscillator.create_ladder_operators(n_cut=n_cut)
    
    if len(circuit.josephson_elements) > 0:
        EJ    = circuit.josephson_elements[0].EJ
        EJ_EC = EJ / EC
        
        # Charge operator
        n = Operator(
            basis_to_matrix={"charge" : np.diag([n for n in range(int(-(n_cut-1)/2), int((n_cut-1)/2) + 1)])}
        )    
        
        n_zpf = (1/2)*(EJ_EC / 2)**(1/4)
    
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
        energy_states = energy_states.astype(complex)
    
        # Unperturbed Hamiltonian operator in the energy basis
        H0["energy"] = np.diag(energies)

        anharmonicity           = (energies[2] - energies[1]) - (energies[1] - energies[0])
        qubit_frequency         = (energies[1] - energies[0]) / h 
        qubit_angular_frequency = qubit_frequency * (2*np.pi)
    
        # Fix eigenvector phases so that <psi_i|n|psi_{i+1}> is negative imaginary
        for i in range(len(energies) - 1):
            element = energy_states[:, i].conj() @ n["charge"] @ energy_states[:, i + 1]
            if np.abs(element) > 1e-12:
                phase = -1j * element / np.abs(element)
                energy_states[:, i + 1] = phase * energy_states[:, i + 1]
            
        # Get n in the energy basis
        n["energy"] = energy_states.conj().T @ n["charge"] @ energy_states
        
    else:
        qubit_angular_frequency = np.sqrt(circuit.omega_squared[0][0])
        
        # Number operator
        N = Operator(
            basis_to_matrix={"fock" : creation["fock"] @ annihilation["fock"]}
        )
        
        n_zpf = (1 / (2*e)) * np.sqrt(hbar * qubit_angular_frequency * C_T / 2)
        
        # Charge operator
        n = Operator(
            basis_to_matrix={"fock" : 1j * n_zpf * (creation["fock"] - annihilation["fock"])}
        )
        
        # Unperturbed Hamiltonian operator
        H0 = Operator(
            basis_to_matrix={"fock": hbar*qubit_angular_frequency*(N["fock"]+(0.5*np.eye(n_cut)))}
        )
        
        # Diagonalize the Hamiltonian in the fock basis to get energy eigenvalues/eigenvectors
        energies, energy_states = np.linalg.eigh(H0["fock"])
        energy_states = energy_states.astype(complex)
        
        # Unperturbed Hamiltonian operator in the energy basis (same as in fock basis)
        H0["energy"] = np.diag(energies)
        
        anharmonicity           = (energies[2] - energies[1]) - (energies[1] - energies[0])
        qubit_frequency         = (energies[1] - energies[0]) / h 
        qubit_angular_frequency = qubit_frequency * (2*np.pi)
        
        # Get n in the energy basis (same as in fock basis)
        n["energy"] = energy_states.conj().T @ n["fock"] @ energy_states

    return n, n_zpf, creation, annihilation, H0, energies, energy_states, anharmonicity, qubit_frequency, qubit_angular_frequency
