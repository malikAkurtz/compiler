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

def main():
    # ---- Shared Hyper-parameters ----
    n_cut              = 41              # Number of charge states, -n_cut : n_cut
    theta              = 0.03            # U_kick angle
    # if basis = "fock", everything will be done in the fock basis
    # if basis = "energy" everything will be done in the energy basis (no fock approximation)
    # i.e. has to be "fock" for Harmonic Oscillator, but acts as a hyperparameter for a Transmon
    basis      = "energy"
    
    # ---- Hyper-parameters for Transmon ----
    EC                 = h * 200 * 1e6   # Charging energy [J]
    EJ_EC              = 50              # EJ/EC ratio
    
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
        basis=basis
    )
    
    # ---- Creating Our Initial Quantum State in Energy/Fock Basis, |0> ----
    probability_amplitudes = (n_cut) * [0]
    probability_amplitudes[0] = 1
    
    probability_amplitudes = np.array(probability_amplitudes)
    probability_amplitudes = probability_amplitudes / np.linalg.norm(probability_amplitudes)
    
    initial_state = Wavefunction(basis_to_coefs={basis : probability_amplitudes})
    
    # ---- Instantiate the Driver ----
    sfq_driver = SFQDriver(
        theta=theta,
        oscillator=transmon,
        basis=basis
    )
        
    # ---- Instantiate System Object ----
    system = System(
        oscillator=transmon,
        sfq_driver=sfq_driver,
        initial_state=initial_state,
        basis=basis
    )
    
    populations = [[] for i in range(n_cut)] # to store measurement probabilities
    
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
        U, U_target = system.RX(np.pi/2)
        U_proj = U[basis][:2, :2]
        
        # leakage = get_leakage(U_proj=U_proj)
        # print(f"Leakage Metric: {leakage}")
        # process_fidelity = get_process_fidelity(U_proj=U_proj, U_target=U_target)
        # print(f"Process Fidelity: {process_fidelity}")
        # avg_gate_fidelity = get_average_gate_fidelity(process_fidelity=process_fidelity, leakage=leakage)
        # print(f"Average Gate Fidelity in the Absence of a Loss Channel: {avg_gate_fidelity}")
        
        probabilities = system.state.get_probabilities(basis)
        
        for idx, p in enumerate(probabilities):
            populations[idx].append(p)
                
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

    plt.ioff()
    plt.show()
        
    
if __name__=="__main__":
    main()