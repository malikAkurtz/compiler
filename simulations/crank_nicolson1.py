import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from circuits.DCSQUIDCircuit import DCSQUIDCircuit
from circuits.TransmonCircuit import TransmonCircuit
from quantum.Matrices import X, Y, Z
from config import *
from circuits.LCResonatorCircuit import LCResonatorCircuit



PLOT = True

def main():
    # ---- Shared Hyper-parameters ----
    n                  = 21            # Number of charge states, -n/2 : n/2 for each transmon
    n_trunc            = 6              # number of states to truncate to for each transmon
    
    # ---- Transmon Circuit Hyper-parameters ----
    fC    = 250e6   # Charging energy frequency EC/h [Hz]
    EJ_EC = 50      # EJ/EC ratio (typical transmon) (higher ratio ==> deeper cosine potential well)
    
    # ---- Derived Physical Constants ----
    EC    = h * fC
    C     = (e**2) / (2 * EC)
    CJ    = 0.05 * C
    CS    = 0.95 * C
    EJ    = EJ_EC * EC          
    
    # ---- Create Ground Node ----
    gnd = Node(branches=[])
    
    # ---- Create Nodes and Branches of Each DCSQUID Circuit (C_JL, C_JR are embedded in C_S) ----
    q1_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=0,
        left_josephson_energy=(EJ/2),
        right_josephson_energy=(EJ/2),
        left_josephson_capacitance=(CJ/2),
        right_josephson_capacitance=(CJ/2),
    )
    
    # ---- Create Transmon Circuit by Adding a Shunt Capacitor Branch ----
    q1 = TransmonCircuit(
        gnd=gnd,
        dcsquid=q1_dcsquid,
        shunt_capacitance=CS,
        coupling_capacitance=0
    )

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = q1.graph
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix = quantize(circuit=circuit, n=n)
    
    system = subsystems[0]

    n_full = n_trunc ** len(subsystems)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_trunc - 1) * [0])})
    o = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_trunc - 2) * [0])})

    # ---- Creating Our Initial Quantum State in energy basis, |0> ----
    state = Wavefunction(basis_to_coefs={"energy" : z["energy"].copy()})
    
    # ---- Transmon Anharmonicity ----
    alpha = system.anharmonicity
    
    # ---- Qubit Frequency ----
    f_q     = system.frequency
    
    # ---- Drive Frequency ----
    f_d     = f_q + OPTIMAL_DETUNING
    
    # ---- Drive period ----
    T_drive = 1 / f_d

    # ---- Total Simulation Time ----
    T       = NUM_PULSES * T_drive
    
    # ---- Delta t ----
    dt      =  T_drive / STEPS_PER_PERIOD
    
    # ---- Total Number of Steps ----
    N_t = int(np.round((T / dt).item()))
    
    # ---- Time Vector ----
    t_vec = np.arange(N_t) * dt.item()
    
    # Result arrays
    Vt_vec = np.zeros(N_t)
    P_0    = np.zeros(N_t)
    P_1    = np.zeros(N_t)
    P_2    = np.zeros(N_t)
    
    # ---- Pulse Amplitude ----
    V_0 = OPTIMAL_AMPLITUDE_SCALE * (system.energies[1] - system.energies[0])
    
    # ---- Pulse Centers ----
    pulse_centers = np.arange(NUM_PULSES) * T_drive

    # ---- Live Plot Setup ----
    update_interval = max(1, N_t // 500)
    if LIVE_VISUALIZATION:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.set_ylabel("Drive Amplitude (GHz)", fontsize=14)
        ax1.set_title("Crank-Nicolson SFQ Drive Simulation", fontsize=16)
        ax1.tick_params(labelsize=12)
        ax2.set_xlabel("Time (ns)", fontsize=14)
        ax2.set_ylabel("Population", fontsize=14)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(labelsize=12)
        line_drive, = ax1.plot([], [], color="tab:blue", linewidth=1.5)
        line_p0,    = ax2.plot([], [], label="|0⟩", linewidth=1.5)
        line_p1,    = ax2.plot([], [], label="|1⟩", linewidth=1.5)
        line_p2,    = ax2.plot([], [], label="|2⟩", linestyle="--", linewidth=1.5)
        ax2.legend(fontsize=13)
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # ---- Crank-Nicolson Loop ----
    for i, t in enumerate(t_vec):
        t_mid = t + (dt / 2)
        
        dt_to_pulses = t_mid - pulse_centers
        mask = np.abs(dt_to_pulses) < 4 * OPTIMAL_SIGMA

        gauss    = np.exp(-0.5 * (dt_to_pulses[mask] / OPTIMAL_SIGMA)**2)
        Vt       = V_0 * np.sum(gauss)
        
        Vt_vec[i] = Vt
        
        # H = H0 + HD(t)
        H_mid = Operator(
            basis_to_matrix={"energy" : system.H0["energy"] + Vt * system.n["energy"]}
        )
        
        A_mat           = np.eye(n_trunc) + (1j * dt / (2 * hbar)) * H_mid["energy"][:n_trunc, :n_trunc]
        B_mat           = np.eye(n_trunc) - (1j * dt / (2 * hbar)) * H_mid["energy"][:n_trunc, :n_trunc]
        state["energy"] = np.linalg.solve(A_mat, B_mat @ state["energy"])

        P_0[i] = state.get_probabilities("energy")[0]
        P_1[i] = state.get_probabilities("energy")[1]
        P_2[i] = state.get_probabilities("energy")[2]

        if LIVE_VISUALIZATION and i % update_interval == 0:
            t_ns_so_far = t_vec[:i+1] * 1e9
            line_drive.set_data(t_ns_so_far, Vt_vec[:i+1] / (h * 1e9))
            line_p0.set_data(t_ns_so_far, P_0[:i+1])
            line_p1.set_data(t_ns_so_far, P_1[:i+1])
            line_p2.set_data(t_ns_so_far, P_2[:i+1])
            ax1.relim(); ax1.autoscale_view()
            ax2.set_xlim(0, t_ns_so_far[-1] if len(t_ns_so_far) > 1 else 1)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # ---- Final Plot ----
    if LIVE_VISUALIZATION:
        plt.ioff()
        plt.savefig("microwave_drive.png", dpi=150)
        plt.show()
    elif PLOT:
        t_ns = t_vec * 1e9
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        ax1.plot(t_ns, Vt_vec / (h * 1e9), color="tab:blue", linewidth=2)
        ax1.set_ylabel("Drive Amplitude (GHz)", fontsize=14)
        ax1.set_title("Crank-Nicolson SFQ Drive Simulation", fontsize=16)
        ax1.tick_params(labelsize=12)
        ax2.plot(t_ns, P_0, label="|0⟩", linewidth=2)
        ax2.plot(t_ns, P_1, label="|1⟩", linewidth=2)
        ax2.plot(t_ns, P_2, label="|2⟩", linestyle="--", linewidth=2)
        ax2.set_xlabel("Time (ns)", fontsize=14)
        ax2.set_ylabel("Population", fontsize=14)
        ax2.legend(fontsize=13)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig("microwave_drive.png", dpi=150)
        plt.show()
    

if __name__=="__main__":
    main()