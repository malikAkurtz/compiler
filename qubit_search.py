import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from Transmon import Transmon
from HarmonicOscillator import HarmonicOscillator
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from itertools import product

PLOT = False


def main():
    # ---- Shared Hyper-parameters ----
    n_cut              = 201             # Number of charge states, -n_cut : n_cut
    n_proj             = 7              # number of states to truncate to
    theta              = 0.03           # U_kick angle
    clock_multiplier   = 8
    ramp_options       = ["00000000", "10000000", "01000000", "00100000","11000000", "10100000", "01100000", "11100000"]
    # if basis = "fock", everything will be done in the fock basis
    # if basis = "energy" everything will be done in the energy basis (no fock approximation)
    # i.e. has to be "fock" for Harmonic Oscillator, but acts as a hyperparameter for a Transmon
    basis              = "energy"
    
    # ---- Hyper-parameters for Transmon ----
    EC                 = h * 250 * 1e6   # Charging energy [J]
    EJ_EC              = 69              # EJ/EC ratio
    
    # ---- Hyper-parameters for Naive Harmonic Oscillator Qubit ----
    C          = 100e-15 # [F]
    L          = 10e-9   # [H]
    
    # Derived Physical Constants
    spring_constant   = 1 / L
    angular_frequency = np.sqrt(spring_constant / C) # [rad/s]
        
    # ---- Create the Oscillator ----
    harmonic_oscillator = HarmonicOscillator(
        mass=C,
        angular_frequency=angular_frequency,
        n_cut=n_cut
    )
    
    transmon = Transmon(
        charging_energy=EC,
        EJ_EC=EJ_EC,
        n_cut=n_cut,
    )
    
    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    probability_amplitudes = (n_proj) * [0]
    probability_amplitudes[0] = 1
    
    probability_amplitudes = np.array(probability_amplitudes)
    probability_amplitudes = probability_amplitudes / np.linalg.norm(probability_amplitudes)
    
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
                
                # ---- Instantiate the Initial State of the Wavefunction ----
                initial_state = Wavefunction(basis_to_coefs={basis : probability_amplitudes})
                
                # ---- Instantiate the Driver ----
                sfq_driver = SFQDriver(
                    theta=theta,
                    oscillator=transmon,
                    basis=basis,
                    ramp=ramp,
                    clock_multiplier=clock_multiplier
                )
                    
                # ---- Instantiate System Object ----
                system = System(
                    clock_multiplier=clock_multiplier,
                    oscillator=transmon,
                    sfq_driver=sfq_driver,
                    initial_state=initial_state,
                    basis=basis,
                    N=N
                )
                
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

                for i in range(1):
                    X_TARGET = np.array([
                        [np.cos(np.pi / 2), -1j * np.sin(np.pi / 2)],
                        [-1j * np.sin(np.pi / 2), np.cos(np.pi / 2)]
                    ])
                    RY_TARGET = np.array([
                        [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
                        [np.sin(theta_target / 2), np.cos(theta_target / 2)]
                    ])
                    RX_TARGET = np.array([
                        [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
                        [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
                    ])
                    H_TARGET = np.array([
                        [1.0, 1.0] / np.sqrt(2),
                        [1.0, -1.0] / np.sqrt(2)
                    ])        
                    
                    TARGET = RY_TARGET
                    
                    system.RY()
                    U = system.state.get_accumulated_unitary()
                    
                    U_proj = U[basis][:2, :2]                    
                    
                    leakage = get_leakage(U_proj=U_proj)
                    process_fidelity = get_process_fidelity(U_proj=U_proj, U_target=TARGET)
                    avg_gate_fidelity = get_average_gate_fidelity(process_fidelity=process_fidelity, leakage=leakage)
                    
                    if avg_gate_fidelity > best_fidelity:
                        best_fidelity = avg_gate_fidelity
                        best_ramp = ramp
                        best_N = N
                        print(f"New Best Fidelity: {avg_gate_fidelity}")
                    
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
                        ax_fock.bar(np.arange(n_cut), probabilities, color='steelblue')
                        ax_fock.set_xlim(-0.5, n_cut - 0.5)
                        ax_fock.set_ylim(0, 1)
                        ax_fock.set_xlabel("Energy (Or Fock) State |n⟩", fontsize=12)
                        ax_fock.set_ylabel("Probability", fontsize=12)
                        ax_fock.set_title(f"Energy (Or Fock) Populations (kick # = {i:.4e} s)", fontsize=14)
                        
                        fig.tight_layout()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
    
    print(f"Best Ramp: {best_ramp}")
    print(f"Best N: {best_N}")
    print(f"Fidelity: {best_fidelity}")
    
    if PLOT:
        plt.ioff()
        plt.show()
        
    
if __name__=="__main__":
    main()