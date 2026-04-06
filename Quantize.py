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
from Operator import Operator
from utils import create_upper_lower

def quantize(circuit: Circuit, n_cut: int):
    """
    Parameters
    ----------
    circuit : Circuit
        A fully constructed Circuit object.
    n_cut : int
        Hilbert-space truncation: Cooper-pair numbers run from -n_cut to +n_cut
        (charge basis), or Fock states 0..n_cut-1 (Fock basis).
    """
    EC_matrix = e**2 / (2 * circuit.capacitance_matrix)
    
    transmons = []
    
    for k in range(len(circuit.josephson_elements)):
        EJ    = circuit.josephson_elements[k]
        EC    = EC_matrix[k][k]
        EJ_EC = EJ / EC
        
        transmons.append(Transmon(
            EC=EC,
            EJ_EC=EJ_EC,
            n_cut=n_cut
        ))
        
    return transmons, EC_matrix
