import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from itertools import product
from Circuit import Circuit
from Quantize import quantize

def main():
    # ---- Shared Hyper-parameters ----
    n_cut              = 201            # Number of charge states, -n_cut : n_cut
    n_proj             = 7              # number of states to truncate to
    clock_multiplier   = 8
    ramp_options       = ["00000000", "10000000", "01000000", "00100000","11000000", "10100000", "01100000", "11100000"]

    
    # ---- Transmon Circuit Hyper-parameters ----
    EJ_EC = 69
    EC    = h * 250 * 1e6   # Charging energy [J]
    theta = 0.03
    
    # ---- Derived Physical Constants ----
    EJ      = EJ_EC * EC                 # [J]
    C_T      = e**2 / (2 * EC)
    
    # ---- Graph Representation of the Circuit ----
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [
            ('a', 'gnd', C_T),  # Shunt + Coupling Capacitance
        ],
        'inductors': [],
        'josephson_elements': [
            ('a', 'gnd', EJ),
        ],
        'external_flux': {}
    }
    
    circuit = Circuit(graph_rep=graph_rep)
    
    n, n_zpf, creation, annihilation, H0, energies, energy_states, alpha, f_q, omega_q = quantize(circuit=circuit, n_cut=n_cut)
    
    print(f"n['energy'][:3, :3]")
    print(n["energy"][:3, :3])
    
    # ---- Create Quantum States |0>, |1>, and the projector onto the computational subspace H_2 ----
    zero_state = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_proj - 1) * [0])})
    one_state  = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_proj - 2) * [0])})
    
    P_Q = Operator(
        basis_to_matrix={"energy": np.outer(to_ket(zero_state["energy"]), to_bra(zero_state["energy"])) + np.outer(to_ket(one_state["energy"]), to_bra(one_state["energy"]))}
    )
    
    # Search over ramps
    best_fidelity = 0
    best_ramp = None
    best_N = None
    
    theta_target = np.pi/2
    
    N_base = int(np.round(theta_target / theta))

    for N in range(N_base - 5, N_base + 5):
        if best_fidelity >= 0.9999:
            break
        if N < 0:
            continue
        for ramp_length in range(1, 6):
            if best_fidelity >= 0.9999:
                break
            for ramp in product(ramp_options, repeat=ramp_length):
                if best_fidelity >= 0.9999:
                    break
                ramp = list(ramp)
                
                # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
                initial_state = Wavefunction(basis_to_coefs={"energy" : zero_state["energy"].copy()})
                    
                # ---- Instantiate System Object ----
                system = System(
                    energy_states=energy_states,
                    n=n,
                    n_zpf=n_zpf,
                    theta=theta,
                    H0=H0,
                    qubit_angular_frequency=omega_q,
                    clock_multiplier=clock_multiplier,
                    initial_state=initial_state,
                    N=N,
                    ramp=ramp
                )
                
                RY_TARGET = Operator(
                    basis_to_matrix={"energy": np.array([
                            [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
                            [np.sin(theta_target / 2), np.cos(theta_target / 2)]
                        ])}
                )
                                    
                system.RY()
                U = system.state.get_accumulated_unitary()
                
                U_Q_full = Operator(
                    basis_to_matrix={"energy" : P_Q["energy"] @ U["energy"] @ P_Q["energy"]}
                )                 
                
                U_Q = Operator(
                    basis_to_matrix={"energy": U_Q_full["energy"][:2, :2]}
                )
                
                L1 = get_L1(U=U_Q, basis="energy")
                
                process_fidelity = get_process_fidelity(U_Q=U_Q, U_target=RY_TARGET, basis="energy")
                avg_gate_fidelity = get_average_gate_fidelity(process_fidelity=process_fidelity, L1=L1)
                
                if avg_gate_fidelity > best_fidelity:
                    best_fidelity = avg_gate_fidelity
                    best_ramp = ramp
                    best_N = N
                    print(f"New Best Fidelity: {avg_gate_fidelity}")
                    print(f"N: {N}, Ramp: {ramp}")
                    
    
    print(f"Best Ramp: {best_ramp}")
    print(f"Best N: {best_N}")
    print(f"Fidelity: {best_fidelity}")
    
if __name__=="__main__":
    main()