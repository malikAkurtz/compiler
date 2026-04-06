import numpy as np
from scipy.linalg import expm
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt

from System import System
from Operator import Operator
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from Circuit import Circuit
from Quantize import quantize

PLOT = True

def main():
    # ---- Shared Hyper-parameters ----
    n_cut              = 201            # Number of charge states, -n_cut : n_cut
    n_proj             = 201              # number of states to truncate to
    clock_multiplier   = 8
    ramp               = ['01000000', '11000000', '10000000', '00000000', '00000000']
    
    # ---- Transmon Circuit Hyper-parameters ----
    EJ_EC = 69
    EC    = h * 250 * 1e6   # Charging energy [J]
    THETA = 0.03
    
    # ---- Derived Physical Constants ----
    EJ    = EJ_EC * EC
    n_zpf = (1/2) * (EJ_EC / 2)**(1/4) # approximation
    BETA  = THETA / (2 * n_zpf)        # approximation
    C_T   = e**2 / (2 * EC)
    CC    = (BETA * hbar * C_T) / FLUX_QUANTUM
    C     = C_T - CC
    
    # ---- Graph Representation of the Transmon Circuit ----
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [
            ('a', 'gnd', C),  # Shunt capacitor
            ('a', 'gnd', CC), # Coupling capacitor
        ],
        'inductors': [],
        'josephson_elements': [
            ('a', 'gnd', EJ),
        ],
        'external_flux': {}
    }
    
    
    circuit = Circuit(graph_rep=graph_rep)
    
    transmons, EC_matrix = quantize(circuit=circuit, n_cut=n_cut)
    
    for k in range(len(transmons)):
        print(f"n['energy'][:2,:2]: ") 
        print(transmons[k].n["energy"][:2,:2])
        print(f"n Hermitian = {np.allclose(transmons[k].n["energy"], transmons[k].n["energy"].conj().T)}")
        print(f"n Unitary = {np.allclose(np.eye(len(transmons[k].n["energy"])), transmons[k].n["energy"] @ transmons[k].n["energy"].conj().T)}")
    
    
    # ---- Create Quantum States |0>, |1>, and the projector onto the computational subspace H_2 ----
    zero_state = Wavefunction(basis_to_coefs={"energy" : np.array([1] + (n_proj - 1) * [0])})
    one_state  = Wavefunction(basis_to_coefs={"energy" : np.array([0] + [1] + (n_proj - 2) * [0])})
    
    P_Q = Operator(
        basis_to_matrix={"energy": np.outer(to_ket(zero_state["energy"]), to_bra(zero_state["energy"])) + np.outer(to_ket(one_state["energy"]), to_bra(one_state["energy"]))}
    )
    
    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : zero_state["energy"].copy()})

    populations = [[] for i in range(n_cut)] # to store measurement probabilities
    
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
            
    # Target Unitary rotation
    theta_target = np.pi/2
    
    # Number of kicks in pulse train
    N = 47
    # N = int(np.round(theta_target / theta))
    
    system = System(
        energy_states=energy_states,
        n=n,
        n_zpf=n_zpf,
        theta=THETA,
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

    for i in range(100):
        system.state.reset_accumulated_unitary() 
        
        system.RY()
        U = system.state.get_accumulated_unitary()

        U_Q_full = Operator(
            basis_to_matrix={"energy" : P_Q["energy"] @ U["energy"] @ P_Q["energy"]}
        )
        
        U_Q = Operator(
            basis_to_matrix={"energy": U_Q_full["energy"][:2, :2]}
        )
        
        L1 = get_L1(U=U_Q, basis="energy")
        
        # print(f"Leakage Metric: {leakage}")
        process_fidelity = get_process_fidelity(U_Q=U_Q, U_target=RY_TARGET, basis="energy")
        # print(f"Process Fidelity: {process_fidelity}")
        avg_gate_fidelity = get_average_gate_fidelity(process_fidelity=process_fidelity, L1=L1)
        # print(f"Average Gate Fidelity in the Absence of a Loss Channel: {avg_gate_fidelity}")
        print(f"L1: {L1}")
        r = np.linalg.norm(get_pauli_coefs(U=U_Q, basis="energy"))
        print(f"r: {r}")
        print(f"Fidelity: {avg_gate_fidelity}")
        
        probabilities = system.state.get_probabilities("energy")
        
        for idx, p in enumerate(probabilities):
            populations[idx].append(p)
        
        if PLOT:
            state_azimuth, state_inclination = get_spherical_coords(alpha=system.state["energy"][0],
                                                                    beta=system.state["energy"][1])
            bx, by, bz = get_rectangular_coords(azimuth=state_azimuth, inclination=state_inclination)
            
            bx_hist.append(bx)
            by_hist.append(by)
            bz_hist.append(bz)
            
            # ---- Bloch Sphere ----
            ax_bloch.cla()
            
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(u.size), np.cos(v))
            ax_bloch.plot_wireframe(x, y, z, alpha=0.08, color='gray', linewidth=0.5)
            
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
            ax_fock.bar(np.arange(n_proj), probabilities, color='steelblue')
            ax_fock.set_xlim(-0.5, n_proj - 0.5)
            ax_fock.set_ylim(0, 1)
            ax_fock.set_xlabel("Energy (Or Fock) State |n⟩", fontsize=12)
            ax_fock.set_ylabel("Probability", fontsize=12)
            ax_fock.set_title(f"Energy (Or Fock) Populations (kick # = {i:.4e} s)", fontsize=14)
            
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

        print(f"Final State: ")
        print(system.state["energy"][:3])
    
    if PLOT:
        plt.ioff()
        plt.show()
        
    
if __name__=="__main__":
    main()