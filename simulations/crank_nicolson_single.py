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
    fC    = 250e6   # Charging energy frequency EC/h [Hz]
    EJ_EC = 50      # EJ/EC ratio (typical transmon) (higher ratio ==> deeper cosine potential well)
    
    # ---- Derived Physical Constants ---- 
    EC    = h * fC
    EJ    = EJ_EC * EC   
    C     = (e**2) / (2 * EC)
    CJ    = 0.05 * C
    CC    = 0.003 * C
    CS    = C - CJ - CC
      
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
        coupling_capacitance=CC
    )

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = q1.graph
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix, C_matrix = quantize(circuit=circuit, n=n)
    
    system = subsystems[0]

    n_full = n_trunc ** len(subsystems)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_trunc - 1) * [0])})
    o = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_trunc - 2) * [0])})

    # ---- Creating Our Initial Quantum State in energy basis, |0> ----
    state = Wavefunction(basis_to_coefs={"energy" : z["energy"].copy()})
    
    ################ ANALYTICAL GUASSIAN MODEL OF SFQ PULSES ################
    
    # ---- Qubit Frequency ----
    f_01    = system.frequency
    
    # ---- Drive Frequency ----
    f_drive = f_01 + OPTIMAL_DETUNING
    
    # ---- Drive period ----
    drive_period = 1 / f_drive
    
    # ---- Total Simulation Time
    total_time = NUM_PULSES * drive_period
    
    # ---- Granularity ----
    dt         =  drive_period / STEPS_PER_PERIOD
    
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

    # Result arrays
    P_0    = np.zeros(len(time))
    P_1    = np.zeros(len(time))
    P_2    = np.zeros(len(time))

    # ---- Live Plot Setup ----
    update_interval = max(1, len(time) // 500)
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
    for i in range(len(time)):
        
        if i == len(time)-1:
            break
        
        v_mid = (Vt_vec[i+1] + Vt_vec[i]) / 2
        
        # H = H0 + HD(t)
        H_mid = Operator(
            basis_to_matrix={"energy" : system.H0["energy"] + ((2*e) * (CC / C) * v_mid) * system.n["energy"]}
        )
        
        A_mat           = np.eye(n_trunc) + (1j * dt / (2 * hbar)) * H_mid["energy"][:n_trunc, :n_trunc]
        B_mat           = np.eye(n_trunc) - (1j * dt / (2 * hbar)) * H_mid["energy"][:n_trunc, :n_trunc]
        state["energy"] = np.linalg.solve(A_mat, B_mat @ state["energy"])

        P_0[i] = state.get_probabilities("energy")[0]
        P_1[i] = state.get_probabilities("energy")[1]
        P_2[i] = state.get_probabilities("energy")[2]

        if LIVE_VISUALIZATION and i % update_interval == 0:
            t_ns_so_far = time[:i+1] * 1e9
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

    t_ns = time * 1e9
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
    plt.savefig("SFQ_drive.png", dpi=150)
    plt.show()
    

if __name__=="__main__":
    main()