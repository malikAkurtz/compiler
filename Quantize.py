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
from Transmon import Transmon
from Circuit import Circuit


def quantize(circuit: Circuit, PHI_off: np.ndarray, PHI_on: np.ndarray, n: int):
    """
    Parameters
    ----------
    circuit : Circuit
        A fully constructed Circuit object.
    n : int
        Hilbert-space dimension: Cooper-pair numbers run from -n to +n
        (charge basis), or Fock states 0..n_cut-1 (Fock basis).
    """
    EC_matrix = (e**2 / 2) * circuit.inv_capacitance_matrix
    
    transmons = []
    
    for k in range(len(circuit.josephson_elements)):
        EJ    = circuit.josephson_elements[k].EJ
        EC    = EC_matrix[k][k]
        EJ_EC = EJ / EC
        
        transmons.append(Transmon(
            EC=EC,
            EJ_EC=EJ_EC,
            PHI_off=PHI_off,
            PHI_on=PHI_on,
            n=n
        ))
        
    return transmons, EC_matrix
