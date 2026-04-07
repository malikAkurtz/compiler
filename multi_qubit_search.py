import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from Transmon import Transmon
from Operator import Operator
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from itertools import product
from Circuit import Circuit
from Quantize import quantize

def main():
    # ---- Shared Hyper-parameters ----
    n                  = 101             # Number of charge states
    n_trunc            = 5              # number of states to truncate to for each transmon
    clock_multiplier   = 8
    ramp_options       = ["00000000", "10000000", "01000000", "00100000",
                          "11000000", "10100000", "01100000", "11100000"]

    # ---- Transmon Circuit Hyper-parameters ----
    THETAS  = np.array([np.pi/100, np.pi/100])
    PHI_on  = np.array([0.130, 0.352, 0.130]) * FLUX_QUANTUM
    PHI_off = np.array([0.130, 0.376, 0.130]) * FLUX_QUANTUM
    
    J1L = 7 * 1e-9  # [nA]
    J2L = 7 * 1e-9  # [nA]
    J1R = 21 * 1e-9 # [nA]
    J2R = 21 * 1e-9 # [nA]
    JcL = 18 * 1e-9 # [nA]
    JcR = 36 * 1e-9 # [nA]
    
    C1  = 70 * 1e-15   # [F]
    C2  = 70 * 1e-15   # [F]
    Cc  = 60 * 1e-15   # [F]
    C12 = 0.25 * 1e-15 # [F]
    C1c = 2 * 1e-15    # [F]
    C2c = 2 * 1e-15    # [F]
    
    C1e = 7.5 * 1e-15  # [F]
    C2e = 7.5 * 1e-15  # [F]
    
    # ---- Graph Representation of the Transmon Circuit ----
    EJ1 = Transmon.calculate_effective_EJ(external_flux=PHI_off[0], JL=J1L, JR=J1R)
    EJc = Transmon.calculate_effective_EJ(external_flux=PHI_off[1], JL=JcL, JR=JcR)
    EJ2 = Transmon.calculate_effective_EJ(external_flux=PHI_off[2], JL=J2L, JR=J2R)
    
    graph_rep = {
        'nodes': ['q1', 'c', 'q2'],
        'capacitors': [
            ('q1', 'gnd', C1),
            ('q1', 'gnd', C1e),
            ('q1', 'c', C1c),
            ('q1', 'q2', C12),
            ('c', 'gnd', Cc),
            ('c', 'q2', C2c),
            ('q2', 'gnd', C2),
            ('q2', 'gnd', C2e),
        ],
        'inductors': [],
        'josephson_elements': [
            ('q1', 'gnd', EJ1),
            ('c',  'gnd', EJc),
            ('q2', 'gnd', EJ2),
        ],
        'external_flux': {}
    }

    circuit = Circuit(graph_rep=graph_rep)
    transmons, EC_matrix = quantize(circuit=circuit, PHI_off=PHI_off, PHI_on=PHI_on, n=n)

    n_full = n_trunc ** len(transmons)
    print(f"Full Hilbert space dimension: {n_full}")

    # ---- Basis states for projector ----
    z = np.array([1] + (n_trunc - 1) * [0])
    o = np.array([0, 1] + (n_trunc - 2) * [0])

    zzz = np.kron(np.kron(z, z), z)  # |000>
    ozz = np.kron(np.kron(o, z), z)  # |100>

    # ---- Projector onto qubit 0 computational subspace ----
    P_0_mat = np.outer(zzz, zzz) + np.outer(ozz, ozz)
    idx_0   = [0, n_trunc**2]  # indices of |000> and |100>

    # ---- Target gate ----
    theta_target = np.pi / 2
    RY_TARGET = Operator(
        basis_to_matrix={"energy": np.array([
            [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
            [np.sin(theta_target / 2),  np.cos(theta_target / 2)]
        ])}
    )

    # ---- Search ----
    best_fidelity = 0
    best_ramp     = None
    best_N        = None

    N_base = int(np.round(theta_target / THETAS[0]))
    print(f"N_base = {N_base}, searching N in [{max(0, N_base-5)}, {N_base+5})")

    # for N in range(max(0, N_base - 5), N_base + 5):
    for N in range(47, 48):
        if best_fidelity >= 0.9999:
            break

        for ramp_length in range(1, 6):
            if best_fidelity >= 0.9999:
                break

            for ramp in product(ramp_options, repeat=ramp_length):
                if best_fidelity >= 0.9999:
                    break

                ramp = list(ramp)

                # ---- Fresh initial state ----
                initial_state = Wavefunction(
                    basis_to_coefs={"energy": np.array([1.0] + [0.0] * (n_full - 1))}
                )

                # ---- Build system ----
                system = System(
                    transmons=transmons,
                    EC_matrix=EC_matrix,
                    thetas=THETAS,
                    clock_multiplier=clock_multiplier,
                    initial_state=initial_state,
                    ramp=ramp,
                    N_kicks=N,
                )

                # ---- Apply gate and get accumulated unitary ----
                system.state.reset_accumulated_unitary()
                system.RY(k=0)
                U = system.state.get_accumulated_unitary()

                # ---- Project onto qubit 0 computational subspace ----
                U_Q0_full = P_0_mat @ U["energy"] @ P_0_mat
                U_Q0_2x2  = Operator(
                    basis_to_matrix={"energy": U_Q0_full[np.ix_(idx_0, idx_0)]}
                )

                # ---- Compute fidelity ----
                L1 = get_L1(U=U_Q0_2x2, basis="energy")
                proc_fid = get_process_fidelity(U_Q=U_Q0_2x2, U_target=RY_TARGET, basis="energy")
                avg_fid  = get_average_gate_fidelity(process_fidelity=proc_fid, L1=L1)

                if avg_fid > best_fidelity:
                    best_fidelity = avg_fid
                    best_ramp     = ramp
                    best_N        = N
                    print(f"New Best Fidelity: {avg_fid:.6f}  |  N={N}, ramp={ramp}")

    print(f"\n===== Search Complete =====")
    print(f"Best Ramp: {best_ramp}")
    print(f"Best N:    {best_N}")
    print(f"Fidelity:  {best_fidelity:.6f}")


if __name__ == "__main__":
    main()