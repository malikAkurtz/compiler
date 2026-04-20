import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import expm
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt

from quantum.System import System
from core.Branch import *
from quantum.Operator import Operator
from quantum.Wavefunction import Wavefunction
from quantum.Matrices import X, Y, Z, CZ
from quantum.utils import *
from core.constants import *
from quantum.fidelity import *
from core.Circuit import Circuit
from core.Quantize import quantize
from circuits.DCSQUIDCircuit import DCSQUIDCircuit
from circuits.TransmonCircuit import TransmonCircuit

PLOT = True

def main():
    # ---- Shared Hyper-parameters ----
    n                  = 51            # Number of charge states, -n/2 : n/2 for each transmon
    n_trunc            = 5              # number of states to truncate to for each transmon
    clock_multiplier   = 8
    ramp               = []
    # ramp               = ['01000000', '11000000', '10000000', '00000000', '00000000']
    
    # ---- Transmon Circuit Hyper-parameters ----
    THETAS  = np.array([np.pi/100, np.pi/100, np.pi/100])
    PHI_off = np.array([0.130, 0.376, 0.130]) * FLUX_QUANTUM
    PHI_on  = np.array([0.130, 0.352, 0.130]) * FLUX_QUANTUM
    
    J_1L = 7 * 1e-9  # [nA]
    J_2L = 7 * 1e-9  # [nA]
    
    J_1R = 21 * 1e-9 # [nA]
    J_2R = 21 * 1e-9 # [nA]
    
    J_CL = 18 * 1e-9 # [nA]
    J_CR = 36 * 1e-9 # [nA]
    
    C_1  = 70 * 1e-15   # [F]
    C_2  = 70 * 1e-15   # [F]
    C_C  = 60 * 1e-15   # [F]
    
    C_12 = 0.25 * 1e-15 # [F]
    C_1C = 2 * 1e-15    # [F]
    C_2C = 2 * 1e-15    # [F]
    
    C_1e = 7.5 * 1e-15  # [F]
    C_2e = 7.5 * 1e-15  # [F]
    
    # ---- Create Ground Node ----
    gnd = Node(label="gnd", branches=[])
    
    # ---- Create Nodes and Branches of Each DCSQUID Circuit (C_JL, C_JR are embedded in self C which comes from TransmonCircuit) ----
    q1_dcsquid = DCSQUID(
        gnd=gnd,
        external_flux=PHI_off[0],
        left_josephson_current=J_1L,
        right_josephson_current=J_1R,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    qc_dcsquid = DCSQUID(
        gnd=gnd,
        external_flux=PHI_off[1],
        left_josephson_current=J_CL,
        right_josephson_current=J_CR,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    q2_dcsquid = DCSQUID(
        gnd=gnd,
        external_flux=PHI_off[2],
        left_josephson_current=J_2L,
        right_josephson_current=J_2R,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )

    dcsquids = [q1_dcsquid, qc_dcsquid, q2_dcsquid]
    
    # ---- Create Transmon Circuits by Adding a Shunt Capacitor Branch to each DCSQUID ----
    q1 = TransmonCircuit(
        dcsquid=q1_dcsquid,
        shunt_capacitance=C_1,
        coupling_capacitance=C_1e
    )
    
    qc = TransmonCircuit(
        dcsquid=qc_dcsquid,
        shunt_capacitance=C_C,
        coupling_capacitance=0
    )
    
    q2 = TransmonCircuit(
        dcsquid=q2_dcsquid,
        shunt_capacitance=C_2,
        coupling_capacitance=C_2e
    )
    
    # ---- Create Branches For Inter-Island Capacitances ----
    cap_12 = Capacitor(capacitance=C_12, nodes=[q1.island, q2.island])
    q1.island.branches.append(cap_12)
    q2.island.branches.append(cap_12)
    
    cap_1C = Capacitor(capacitance=C_1C, nodes=[q1.island, qc.island])
    q1.island.branches.append(cap_1C)
    qc.island.branches.append(cap_1C)
    
    cap_2C = Capacitor(capacitance=C_2C, nodes=[q2.island, qc.island])
    q2.island.branches.append(cap_2C)
    qc.island.branches.append(cap_2C)

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = Graph(
        vertices=[gnd, q1.island, qc.island, q2.island], 
        edges=q1.branches + qc.branches + q2.branches
        )
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    transmons, EC_matrix = quantize(circuit=circuit, n=n)
        
    n_full = n_trunc ** len(transmons)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_trunc - 1) * [0])})
    o = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_trunc - 2) * [0])})
    
    # ---- Create Full Subsystem Quantum Basis States ----
    zzz     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(z["energy"], z["energy"]), z["energy"])}) # |000>
    zzo     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(z["energy"], z["energy"]), o["energy"])}) # |001>
    ozz     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(o["energy"], z["energy"]), z["energy"])}) # |100>
    ozo     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(o["energy"], z["energy"]), o["energy"])}) # |101>
    
    basis_states = [zzz, zzo, ozz, ozo]
    
    # ---- Creating Our Initial Quantum State in Energy basis ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : zzz["energy"].copy()})
    
    # ---- Creating Our Quantum System ----
    system = System(
        transmons=transmons,
        dcsquids=dcsquids,
        EC_matrix=EC_matrix,
        thetas=THETAS,
        clock_multiplier=clock_multiplier,
        initial_state=initial_state,
        ramp=ramp,
        PHI_off=PHI_off,
        PHI_on=PHI_on
    )
    
    # ---- Setup for Time-Series Data ----
    times = []
    thetas_t = []
    phis_t = []
    
    dt = 1e-9 # 1ns step
    total_steps = 100

    for step in range(total_steps):
        system.state.reset_accumulated_unitary() 
        
        system.fSim(duration=step * dt)
        
        # (n_full x n_full)
        U = system.state.get_accumulated_unitary()

        # (n_full x n_full)
        # P|psi> = alpha|000> + beta|100> + gamma|001> + eta|101>
        P = Operator(
            basis_to_matrix={
                "energy": np.outer(to_ket(zzz["energy"]), to_bra(zzz["energy"])) +
                        np.outer(to_ket(ozz["energy"]), to_bra(ozz["energy"])) +
                        np.outer(to_ket(zzo["energy"]), to_bra(zzo["energy"])) +
                        np.outer(to_ket(ozo["energy"]), to_bra(ozo["energy"]))
            }
        )
    
        # (n_full x n_full)
        U_Q = Operator(
            basis_to_matrix={"energy": P["energy"] @ U["energy"] @ P["energy"]}
        )
    
        # (4 x 4)
        U_Q_4x4_matrix = np.zeros((4, 4), dtype=complex)
        for i in range(len(basis_states)):
            for j in range(len(basis_states)):
                U_Q_4x4_matrix[i][j] = (to_bra(basis_states[i]["energy"]) @ U_Q["energy"] @ to_ket(basis_states[j]["energy"])).item()
            
        U_Q_4x4 = Operator(
            basis_to_matrix={"energy" : U_Q_4x4_matrix}
        )
        
        a = U_Q_4x4["energy"][1, 1]
        c = U_Q_4x4["energy"][3, 3]
        
        theta = np.real(np.arccos(np.clip(np.real(a), -1.0, 1.0)))
        phi   = np.angle(c)
        
        times.append(step * dt)
        thetas_t.append(theta)
        phis_t.append(phi)
    
    print(thetas_t[17])
    print(phis_t[17])
    
    print(f"time: {times[17]}")
    print(f"theta @ 17 ns: {thetas_t[17]}")
    print(f"phi @ 17 ns: {phis_t[17]}")
        
    # ---- Plotting the Results ----
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, thetas_t, label=r'$\theta(t)$', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('iSWAP Interaction Angle')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, phis_t, label=r'$\phi(t)$', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Phi (rad)')
    plt.title('CPHASE Interaction Angle')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    main()