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



PLOT = True

def main():
    # ---- Shared Hyper-parameters ----
    n                  = 201            # Number of charge states, -n/2 : n/2 for each transmon
    n_trunc            = 7              # number of states to truncate to for each transmon
    THETAS             = np.array([0.03])
    clock_multiplier   = 8
    ramp               = ['01000000', '11000000', '10000000', '00000000', '00000000']
    # ramp               = []
    
    # ---- Transmon Circuit Hyper-parameters ----
    EJ_EC = 69
    EC    = h * 250 * 1e6   # Charging energy [J]
    
    # ---- Derived Physical Constants ----
    EJ    = EJ_EC * EC
    n_zpf = (1/2) * (EJ_EC / 2)**(1/4) # approximation
    BETAS = THETAS / (2 * n_zpf)        # approximation
    C     = e**2 / (2 * EC)            # includes Josephson Capacitance CJ
    C_C   = (BETAS[0] * hbar * C) / FLUX_QUANTUM
    C_S   = C - C_C
    
    # ---- Create Ground Node ----
    gnd = Node(branches=[])
    
    # ---- Create Nodes and Branches of Each DCSQUID Circuit (C_JL, C_JR are embedded in C_S) ----
    q1_dcsquid = DCSQUIDCircuit(
        gnd=gnd,
        external_flux=0,
        left_josephson_energy=EJ,
        right_josephson_energy=EJ,
        left_josephson_capacitance=0,
        right_josephson_capacitance=0,
    )
    
    # ---- Create Transmon Circuit by Adding a Shunt Capacitor Branch ----
    q1 = TransmonCircuit(
        gnd=gnd,
        dcsquid=q1_dcsquid,
        shunt_capacitance=C_S,
        coupling_capacitance=C_C
    )

    # ---- Create the Larger Circuit Graph Object (G = (V, E)) ----
    circuit_graph = Graph(vertices=[gnd, q1.island], edges=q1.branches)
    
    # ---- Create Circuit Object From Graph ----
    circuit = Circuit(circuit_graph)
    
    # ---- Build Matrices ----
    circuit.build()
    
    subsystems, EC_matrix = quantize(circuit=circuit, n=n)
        
    n_full = n_trunc ** len(subsystems)
    
    print(f"Full Hilbert space dimension: {n_full}")
    
    # ---- Create individual Subsystem Quantum Basis States ----
    z = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_trunc - 1) * [0])})
    o = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_trunc - 2) * [0])})

    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : z["energy"].copy()})
    
    if PLOT:
        plt.ion()
        fig = plt.figure(figsize=(16, 8))

        # ---- Bloch Sphere Subplot ----
        ax_bloch = fig.add_subplot(2, 3, 1, projection='3d')

        # ---- Projection Subplots ----
        ax_xy = fig.add_subplot(2, 3, 2)
        ax_xz = fig.add_subplot(2, 3, 3)
        ax_yz = fig.add_subplot(2, 3, 4)

        # ---- Fock Populations Subplot ----
        ax_fock = fig.add_subplot(2, 3, (5, 6))

        # Store Bloch vector history for trail
        bx_hist, by_hist, bz_hist = [], [], []
    
    system = System(
        subsystems=subsystems,
        dcsquids=[q1_dcsquid],
        EC_matrix=EC_matrix,
        thetas=THETAS,
        clock_multiplier=clock_multiplier,
        initial_state=initial_state,
        ramp=ramp,
        PHI_off=[], 
        PHI_on=[]
    )
    
    # Target Unitary rotation
    theta_target = np.pi/2
    
    # Qubit to rotate
    k = 0
    
    U_TARGET = get_RY_target(theta_target)

    for i in range(1):
        system.state.reset_accumulated_unitary() 
        
        system.RY(k=k, theta_target=theta_target)
        
        # (n_full x n_full)
        U = system.state.get_accumulated_unitary()
        
        # NOTE: The logical bit strings are in base n_trunc, i.e.
        # |b2, b1, b0> = b2 * (n_trunc**2) + b1 * (n_trunc**1) + b0 * (n_trunc**0)
        # len(|b2, b1, b0>) = n_trunc**3

        
         # ---- Calculate Gate Fidelity for Qubit 0 ----
        # Want to project onto the first k=0 qubit computational subspace
        # Want P|psi> = a|0> + b|1>
        # (n_full x n_full)
        P_0 = Operator(
            basis_to_matrix={
                "energy": np.outer(to_ket(z["energy"]), to_bra(z["energy"])) + \
                            np.outer(to_ket(o["energy"]), to_bra(o["energy"]))
                }
        )
        # (n_full x n_full)
        U_Q0 = Operator(
            basis_to_matrix={"energy": P_0["energy"] @ U["energy"] @ P_0["energy"]}
        )
        idx_0 = [0, n_trunc**0]
        # (2 x 2)
        U_Q0_2x2 = Operator(
            basis_to_matrix={"energy": U_Q0["energy"][np.ix_(idx_0, idx_0)]}
        )
        L1_0 = get_L1(U=U_Q0_2x2, basis="energy")
        process_fidelity_0 = get_process_fidelity(U_Q=U_Q0_2x2, U_target=U_TARGET, basis="energy")
        avg_gate_fidelity_0 = get_average_gate_fidelity(process_fidelity=process_fidelity_0, L1=L1_0)
        r_0 = np.linalg.norm(get_pauli_coefs(U=U_Q0_2x2, basis="energy"))
        print(f"Gate on Qubit {0} Fidelity: {avg_gate_fidelity_0}")
        
        probabilities = system.state.get_probabilities("energy")
        
        rho = np.outer(system.state["energy"], system.state["energy"].conj())
        
        bx = np.trace(rho[:2, :2] @ X).real
        by = np.trace(rho[:2, :2] @ Y).real
        bz = np.trace(rho[:2, :2] @ Z).real
        
        
        if PLOT:
            
            bx_hist.append(bx)
            by_hist.append(by)
            bz_hist.append(bz)
            
            # ---- Bloch Sphere ----
            ax_bloch.cla()
            
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x_mesh = np.outer(np.cos(u), np.sin(v))
            y_mesh = np.outer(np.sin(u), np.sin(v))
            z_mesh = np.outer(np.ones(u.size), np.cos(v))
            ax_bloch.plot_wireframe(x_mesh, y_mesh, z_mesh, alpha=0.08, color='gray', linewidth=0.5)
            
            ax_bloch.plot([-1, 1], [0, 0], [0, 0], color='gray', linewidth=0.5, linestyle='--')
            ax_bloch.plot([0, 0], [-1, 1], [0, 0], color='gray', linewidth=0.5, linestyle='--')
            ax_bloch.plot([0, 0], [0, 0], [-1, 1], color='gray', linewidth=0.5, linestyle='--')
            
            ax_bloch.text(0, 0, 1.15, "|0⟩", ha='center', fontsize=12, fontweight='bold')
            ax_bloch.text(0, 0, -1.15, "|1⟩", ha='center', fontsize=12, fontweight='bold')
            ax_bloch.text(1.15, 0, 0, "X", ha='center', fontsize=10, color='gray')
            ax_bloch.text(0, 1.15, 0, "Y", ha='center', fontsize=10, color='gray')
            
            ax_bloch.quiver(0, 0, 0, bx, by, bz, color='red', arrow_length_ratio=0.08, linewidth=2.5)
            ax_bloch.scatter([bx], [by], [bz], color='red', s=40, zorder=5)
            
            ax_bloch.set_xlim([-1.3, 1.3])
            ax_bloch.set_ylim([-1.3, 1.3])
            ax_bloch.set_zlim([-1.3, 1.3])
            ax_bloch.set_box_aspect([1, 1, 1])
            ax_bloch.set_title("Bloch Sphere", fontsize=14, pad=10)
            ax_bloch.set_axis_off()
            ax_bloch.view_init(elev=20, azim=30)
            
            # ---- XY Projection ----
            ax_xy.cla()
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=0.5)
            ax_xy.add_patch(circle)
            ax_xy.plot(bx_hist, by_hist, color='blue', alpha=0.3, linewidth=1)
            ax_xy.scatter([bx], [by], color='red', s=50, zorder=5)
            ax_xy.axhline(0, color='gray', linewidth=0.3)
            ax_xy.axvline(0, color='gray', linewidth=0.3)
            ax_xy.set_xlim([-1.3, 1.3])
            ax_xy.set_ylim([-1.3, 1.3])
            ax_xy.set_aspect('equal')
            ax_xy.set_xlabel("X")
            ax_xy.set_ylabel("Y")
            ax_xy.set_title("XY Projection", fontsize=12)
            
            # ---- XZ Projection ----
            ax_xz.cla()
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=0.5)
            ax_xz.add_patch(circle)
            ax_xz.plot(bx_hist, bz_hist, color='blue', alpha=0.3, linewidth=1)
            ax_xz.scatter([bx], [bz], color='red', s=50, zorder=5)
            ax_xz.axhline(0, color='gray', linewidth=0.3)
            ax_xz.axvline(0, color='gray', linewidth=0.3)
            ax_xz.set_xlim([-1.3, 1.3])
            ax_xz.set_ylim([-1.3, 1.3])
            ax_xz.set_aspect('equal')
            ax_xz.set_xlabel("X")
            ax_xz.set_ylabel("Z")
            ax_xz.set_title("XZ Projection", fontsize=12)
            
            # ---- YZ Projection ----
            ax_yz.cla()
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=0.5)
            ax_yz.add_patch(circle)
            ax_yz.plot(by_hist, bz_hist, color='blue', alpha=0.3, linewidth=1)
            ax_yz.scatter([by], [bz], color='red', s=50, zorder=5)
            ax_yz.axhline(0, color='gray', linewidth=0.3)
            ax_yz.axvline(0, color='gray', linewidth=0.3)
            ax_yz.set_xlim([-1.3, 1.3])
            ax_yz.set_ylim([-1.3, 1.3])
            ax_yz.set_aspect('equal')
            ax_yz.set_xlabel("Y")
            ax_yz.set_ylabel("Z")
            ax_yz.set_title("YZ Projection", fontsize=12)
            
            # ---- Fock Populations ----
            ax_fock.cla()
            ax_fock.bar(np.arange(n_full), probabilities, color='steelblue')
            ax_fock.set_xlim(-0.5, n_full - 0.5)
            ax_fock.set_ylim(0, 1)
            ax_fock.set_xlabel("Energy (Or Fock) State |n⟩", fontsize=12)
            ax_fock.set_ylabel("Probability", fontsize=12)
            ax_fock.set_title(f"Energy (Or Fock) Populations (kick # = {i:.4e} s)", fontsize=14)
            
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

        print(f"Final State: ")
        print(system.state["energy"])
    
    if PLOT:
        plt.ioff()
        plt.show()
        
    
if __name__=="__main__":
    main()