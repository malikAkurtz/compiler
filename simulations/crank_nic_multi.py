import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantum.Operator import Operator
from quantum.Wavefunction import Wavefunction
from quantum.utils import *
from core.constants import *
from quantum.fidelity import *
from core.Circuit import Circuit
from core.Quantize import quantize
from core.Branch import *
from circuits.DCSQUIDCircuit import DCSQUIDCircuit
from circuits.TransmonCircuit import TransmonCircuit
from config import *
from quantum.System import System

def main():
    # ---- Shared Hyper-parameters ----
    n                  = 21            # Number of charge states, -n/2 : n/2 for each transmon
    n_trunc            = 6              # number of states to truncate to for each transmon
    
    # ---- Transmon Circuit Hyper-parameters ----
    PHI_off = np.array([0.130, 0.376, 0.130]) * FLUX_QUANTUM
    PHI_on  = np.array([0.130, 0.352, 0.130]) * FLUX_QUANTUM
    
    J_1L = 7 * 1e-9  # [nA]
    J_2L = 5 * 1e-9  # [nA]
    
    J_1R = 21 * 1e-9 # [nA]
    J_2R = 15 * 1e-9 # [nA]
    
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
    
    # ---- Derived Phyiscal Constants ----
    EJ_1L = J_1L * REDUCED_FLUX_QUANTUM
    EJ_2L = J_2L * REDUCED_FLUX_QUANTUM
    
    EJ_1R = J_1R * REDUCED_FLUX_QUANTUM
    EJ_2R = J_2R * REDUCED_FLUX_QUANTUM
    
    EJ_CL = J_CL * REDUCED_FLUX_QUANTUM
    EJ_CR = J_CR * REDUCED_FLUX_QUANTUM
    
      
   # ---- Create Ground Node ----
    gnd = Node(branches=[])
    
    # ---- Create Nodes and Branches of Each DCSQUID Circuit (C_JL, C_JR are embedded in self C which comes from TransmonCircuit) ----
    q1_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=PHI_off[0],
        left_josephson_energy=EJ_1L,
        right_josephson_energy=EJ_1R,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    qc_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=PHI_off[1],
        left_josephson_energy=EJ_CL,
        right_josephson_energy=EJ_CR,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    q2_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=PHI_off[2],
        left_josephson_energy=EJ_2L,
        right_josephson_energy=EJ_2R,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )

    # ---- Create Transmon Circuits by Adding a Shunt Capacitor Branch to each DCSQUID ----
    q1 = TransmonCircuit(
        gnd=gnd,
        dcsquid=q1_dcsquid,
        shunt_capacitance=C_1,
        coupling_capacitance=C_1e
    )
    
    qc = TransmonCircuit(
        gnd=gnd,
        dcsquid=qc_dcsquid,
        shunt_capacitance=C_C,
        coupling_capacitance=0
    )
    
    q2 = TransmonCircuit(
        gnd=gnd,
        dcsquid=q2_dcsquid,
        shunt_capacitance=C_2,
        coupling_capacitance=C_2e
    )
    
    transmons = [q1, qc, q2]
    
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
        edges=q1.branches + qc.branches + q2.branches + [cap_12] + [cap_1C] + [cap_2C]
        )
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix, C_matrix = quantize(circuit=circuit, n=n)
    
    n_full = n_trunc ** len(subsystems)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_trunc - 1) * [0])})
    o = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_trunc - 2) * [0])})
    
    # ---- Create Full Subsystem Quantum Basis States ----
    zzz     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(z["energy"], z["energy"]), z["energy"])}) # |000>
    zzo     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(z["energy"], z["energy"]), o["energy"])}) # |001>
    ozz     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(o["energy"], z["energy"]), z["energy"])}) # |100>
    ozo     = Wavefunction(basis_to_coefs={"energy" : np.kron(np.kron(o["energy"], z["energy"]), o["energy"])}) # |101>
    
    # ---- Creating Our Initial Quantum State in Energy basis ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : zzz["energy"].copy()})
    
    # ---- Creating Composite System Object ----
    system = System(
        subsystems=subsystems,
        transmons=transmons,
        EC_matrix=EC_matrix,
        C_matrix=C_matrix,
        initial_state=initial_state,
        PHI_off=PHI_off,
        PHI_on=PHI_on
    )
    
    # ---- The qubit to evolve
    k = 0
    
    ################ ANALYTICAL GUASSIAN MODEL OF SFQ PULSES ################
    
    # ---- Qubit Frequency ----
    f_01 = system.dressed_frequencies[k]
    
    # ---- Drive Frequency ----
    f_drive = f_01 + OPTIMAL_DETUNING
    
    # ---- Drive period ----
    drive_period = 1 / f_drive
    
    # ---- Total Simulation Time
    total_time = NUM_PULSES * drive_period
    
    # ---- Granularity ----
    dt =  drive_period / STEPS_PER_PERIOD
    
    # ---- Total Time Steps in Simulation ----
    total_time_steps = int(np.round((total_time / dt).item()))
    
    # ---- Time Vector ----
    time = np.arange(total_time_steps) * dt.item()
    
    # Physical pulse amplitude: Gaussian with area = Phi_0
    Phi0 = h / (2 * e)
    V_0 = Phi0 / (OPTIMAL_SIGMA * np.sqrt(2 * np.pi))

    Vt_vec = generate_sfq_pulses(
        steps_per_period=STEPS_PER_PERIOD,
        num_pulses=NUM_PULSES,
        amplitude_scale=V_0,
        drive_period=drive_period,
        pulse_width=OPTIMAL_SIGMA
    )
    
    ########################################################################
    
    ################ V(t) LOOKUP TABLE PROVIDED BY RIYA ####################
    
    # # ---- Import Time-Voltage CSV
    # Vt_df = pd.read_csv("sfq_V_lookup.csv")
    
    # time = Vt_df["time_s"].to_numpy()
    
    # dt = time[1] - time[0]
    
    # Vt_vec = Vt_df["voltage_V"].to_numpy()
    
    ########################################################################
    

    # ---- Crank-Nicolson ----
    P = system.crank_nicolson(
        time=time,
        Vt1_vec=Vt_vec,
        Vt2_vec=np.zeros(len(time)),
    )
    
    t_ns = time * 1e9
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(t_ns, Vt_vec * 1e6, color="tab:blue", linewidth=2)
    ax1.set_ylabel("Voltage (μV)", fontsize=14)
    ax1.set_title("Crank-Nicolson Multi-Qubit SFQ Drive", fontsize=16)
    ax2.plot(t_ns, P[:, 0], label="|0⟩", linewidth=2)
    ax2.plot(t_ns, P[:, 1], label="|1⟩", linewidth=2)
    ax2.plot(t_ns, P[:, 2], label="|2⟩", linestyle="--", linewidth=2)
    ax2.set_xlabel("Time (ns)", fontsize=14)
    ax2.set_ylabel("Qubit 1 Population", fontsize=14)
    ax2.legend(fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("crank_nicolson_multi.png", dpi=150)
    plt.show()
    
    
        

if __name__=="__main__":
    main()