import numpy as np
from scipy.linalg import expm
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt

from System import System
from Branch import *
from Operator import Operator
from Wavefunction import Wavefunction
from Matrices import X, Y, Z
from utils import *
from constants import *
from fidelity import *
from Circuit import Circuit
from Quantize import quantize
from DCSQUID import DCSQUID
from TransmonCircuit import TransmonCircuit

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
    
    # ---- Creating Our Initial Quantum State in Energy basis ----
    initial_state = Wavefunction(basis_to_coefs={"energy" : zzz["energy"].copy()})
    
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
    
    # Target Unitary rotation
    theta_target = np.pi/2
    
    # Qubit to rotate
    k = 0
    
    # Target Unitary on the computational subspace of a single qubit
    U_TARGET = get_RX_target(theta_target)

    for i in range(1):
        system.state.reset_accumulated_unitary() 
        
        system.RX(k, theta_target)
        
        # (n_full x n_full)
        U = system.state.get_accumulated_unitary()

        # NOTE: The logical bit strings are in base n_trunc, i.e.
        # |b2, b1, b0> = b2 * (n_trunc**2) + b1 * (n_trunc**1) + b0 * (n_trunc**0)
        # len(|b2, b1, b0>) = n_trunc**3
        
        # ---- Calculate Gate Fidelity for Qubit 0 ----
        # Want to project onto the first k=0 qubit computational subspace
        # Want P|psi> = a|000> + b|100>
        # (n_full x n_full)
        P_0 = Operator(
            basis_to_matrix={
                "energy": np.outer(to_ket(zzz["energy"]), to_bra(zzz["energy"])) + \
                            np.outer(to_ket(ozz["energy"]), to_bra(ozz["energy"]))
                }
        )
        # (n_full x n_full)
        U_Q0 = Operator(
            basis_to_matrix={"energy": P_0["energy"] @ U["energy"] @ P_0["energy"]}
        )
        idx_0 = [0, n_trunc**2]
        # (2 x 2)
        U_Q0_2x2 = Operator(
            basis_to_matrix={"energy": U_Q0["energy"][np.ix_(idx_0, idx_0)]}
        )
        L1_0 = get_L1(U=U_Q0_2x2, basis="energy")
        process_fidelity_0 = get_process_fidelity(U_Q=U_Q0_2x2, U_target=U_TARGET, basis="energy")
        avg_gate_fidelity_0 = get_average_gate_fidelity(process_fidelity=process_fidelity_0, L1=L1_0)
        r_0 = np.linalg.norm(get_pauli_coefs(U=U_Q0_2x2, basis="energy"))
        print(f"Gate on Qubit {0} Fidelity: {avg_gate_fidelity_0}")
        
        # ---- Calculate Gate Fidelity for Qubit 2 ----
        # Want to project onto the second k=2 qubit computational subspace
        # Want P|psi> = a|000> + b|001>
        # (n_full x n_full)
        P_2 = Operator(
            basis_to_matrix={
                "energy": np.outer(to_ket(zzz["energy"]), to_bra(zzz["energy"])) + \
                            np.outer(to_ket(zzo["energy"]), to_bra(zzo["energy"]))
                }
        )
        # (n_full x n_full)
        U_Q2 = Operator(
            basis_to_matrix={"energy": P_2["energy"] @ U["energy"] @ P_2["energy"]}
        )
        idx_2 = [0, n_trunc**0]
        # (2 x 2)
        U_Q2_2x2 = Operator(
            basis_to_matrix={"energy": U_Q2["energy"][np.ix_(idx_2, idx_2)]}
        )
        L1_2 = get_L1(U=U_Q2_2x2, basis="energy")
        process_fidelity_2 = get_process_fidelity(U_Q=U_Q2_2x2, U_target=Operator(basis_to_matrix={"energy": np.eye(2)}), basis="energy")
        avg_gate_fidelity_2 = get_average_gate_fidelity(process_fidelity=process_fidelity_2, L1=L1_2)
        r_2 = np.linalg.norm(get_pauli_coefs(U=U_Q2_2x2, basis="energy"))
        print(f"Gate on Qubit {2} Fidelity: {avg_gate_fidelity_2}")
        
        probabilities = system.state.get_probabilities("energy")
                
        rho = np.outer(to_ket(system.state["energy"]), to_bra(system.state["energy"]))
        
        psi = system.state["energy"].reshape(n_trunc, n_trunc, n_trunc)
        
        A = psi.reshape(n_trunc, n_trunc**2)
        
        A = A.reshape(n_trunc, n_trunc**2)
        
        rho_1 = A @ A.conj().T
        
        bx = np.trace(rho_1[:2, :2] @ X).real
        by = np.trace(rho_1[:2, :2] @ Y).real
        bz = np.trace(rho_1[:2, :2] @ Z).real
        
        if PLOT:
            
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
        print(system.state["energy"][:3])
    
    if PLOT:
        plt.ioff()
        plt.show()
        
    
if __name__=="__main__":
    main()