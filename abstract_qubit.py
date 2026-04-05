import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from Transmon import Transmon
from HarmonicOscillator import HarmonicOscillator
from SFQDriver import SFQDriver
from Operator import Operator
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from itertools import product
from Circuit import Circuit
from Quantize import quantize

PLOT = True

def main():
    # ---- Shared Hyper-parameters ----
    n_cut              = 201            # Number of charge states, -n_cut : n_cut
    n_proj             = 7              # number of states to truncate to
    clock_multiplier   = 8
    ramp               = ['11000000', '10100000', '00000000', '00000000']
    # ramp               = []
    # if basis = "fock", everything will be done in the fock basis
    # if basis = "energy" everything will be done in the energy basis (no fock approximation)
    # i.e. has to be "fock" for Harmonic Oscillator, but acts as a hyperparameter for a Transmon
    basis              = "energy"
    
    # ---- Transmon Circuit Hyper-parameters ----
    EJ_EC = 69
    EC    = h * 250 * 1e6   # Charging energy [J]
    theta = 0.03
    
    # ---- Instantiate the Transmon object ----
    transmon = Transmon(
        charging_energy=EC,
        EJ_EC=EJ_EC,
        n_cut=n_cut,
    )
    
    # ---- Derived Physical Constants ----
    omega_q = transmon.angular_frequency # [rad/sec]
    EJ      = EJ_EC * EC                 # [J]
    a       = theta / (FLUX_QUANTUM * np.sqrt(2 * omega_q))
    C_T      = e**2 / (2 * EC)
    C       = ((-a + np.sqrt(a**2 + 4*C_T)) / 2)**2
    CC      = C_T - C
    
    # ---- Graph Representation of the Circuit ----
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [
            ('a', 'gnd', C),  # Shunt Capacitance
            ('a', 'gnd', CC), # Coupling Capacitance
        ],
        'inductors': [],
        'josephson_elements': [
            ('a', 'gnd', EJ),
        ],
        'external_flux': {}
    }
    
    circuit = Circuit(graph_rep=graph_rep)
    
    n, n_zpf, creation, annihilation, H0, energies, energy_states, alpha, f_q, omega_q = quantize(circuit=circuit, n_cut=n_cut)
    
    # ---- Create Quantum States |0>, |1>, and the projector onto the computational subspace H_2 ----
    zero_state = Wavefunction(basis_to_coefs={basis : np.array([1] + (n_proj - 1) * [0])})
    one_state  = Wavefunction(basis_to_coefs={basis : np.array([0] + [1] + (n_proj - 2) * [0])})
    
    P_Q = Operator(
        basis_to_matrix={basis: np.outer(to_ket(zero_state[basis]), to_bra(zero_state[basis])) + np.outer(to_ket(one_state[basis]), to_bra(one_state[basis]))}
    )
    
    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    initial_state = Wavefunction(basis_to_coefs={basis : zero_state[basis].copy()})

    # ---- Create the Driving Hamiltonian Operator ----
    HD = Operator(
        basis_to_matrix={}
    )
    
    if basis == "energy":
        theta_prime  = theta / n_zpf
        HD["energy"] = expm( (-1j * (theta_prime) / 2) * n["fock"] )
    else:
        HD["fock"]   = expm( (theta/2) * (creation["fock"] - annihilation["fock"]) )

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
    
    system = System(
        clock_multiplier=clock_multiplier,
        oscillator=transmon,
        sfq_driver=sfq_driver,
        initial_state=initial_state,
        basis=basis,
        N=N
    )
    
    RY_TARGET = Operator(
        basis_to_matrix={basis: np.array([
                [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
                [np.sin(theta_target / 2), np.cos(theta_target / 2)]
            ])}
    )

    for i in range(1):
                
        system.state.reset_accumulated_unitary() 
        
        system.RY()
        U = system.state.get_accumulated_unitary()

        U_Q_full = Operator(
            basis_to_matrix={basis : P_Q[basis] @ U[basis] @ P_Q[basis]}
        )
        
        U_Q = Operator(
            basis_to_matrix={basis: U_Q_full[basis][:2, :2]}
        )
        
        L1 = get_L1(U=U_Q, basis=basis)
        
        # print(f"Leakage Metric: {leakage}")
        process_fidelity = get_process_fidelity(U_Q=U_Q, U_target=RY_TARGET, basis=basis)
        # print(f"Process Fidelity: {process_fidelity}")
        avg_gate_fidelity = get_average_gate_fidelity(process_fidelity=process_fidelity, L1=L1)
        # print(f"Average Gate Fidelity in the Absence of a Loss Channel: {avg_gate_fidelity}")
        print(f"L1: {L1}")
        r = np.linalg.norm(get_pauli_coefs(U=U_Q, basis=basis))
        print(f"r: {r}")
        print(f"Fidelity: {avg_gate_fidelity}")
        
        probabilities = system.state.get_probabilities(basis)
        
        for idx, p in enumerate(probabilities):
            populations[idx].append(p)
        
        if PLOT:
            state_azimuth, state_inclination = get_spherical_coords(alpha=system.state[basis][0],
                                                                    beta=system.state[basis][1])
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
        print(system.state[basis][:3])
    
    if PLOT:
        plt.ioff()
        plt.show()
        
    
if __name__=="__main__":
    main()