import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


from constants import hbar
from HarmonicOscillator import HarmonicOscillator
from Wavefunction import Wavefunction
from utils import *

def main():
    # ---- Hyper-parameters for Harmonic Oscillator ----
    C     = 100     # [fF]
    L     = 10      # [nH]
    n_cut = 20      # Number of Fock states to truncate to
    
    # ---- Derived Physical Constants ----
    k = 1 / L
    m = C
    omega = 2 * np.pi * np.sqrt(k / m) # [rad/s]
    
    harmonic_oscillator = HarmonicOscillator(mass=m, omega=omega, n_cut=n_cut)
    
    print(f"Creation and Annihilation Operators in Fock/Energy basis: ")
    print(f"a: ")
    print(harmonic_oscillator.a)
    print(f"a_dagger: ")
    print(harmonic_oscillator.a_dagger)
    print(f"Number Operator n in Fock/Energy basis: ")
    print(harmonic_oscillator.n)
    print("Hamiltonian in Fock/Energy basis: ")
    print(harmonic_oscillator.H)
    
    # ---- Creating Our Initial Quantum State ----
    probability_amplitudes = n_cut * [1]
    # probability_amplitudes[0] = 1 / np.sqrt(2) 
    # probability_amplitudes[1] = 1 / np.sqrt(2) 
    probability_amplitudes = np.array(probability_amplitudes)
    probability_amplitudes = probability_amplitudes / np.linalg.norm(probability_amplitudes)
    
    initial_state = Wavefunction(probability_amplitudes=probability_amplitudes, basis=harmonic_oscillator.H.basis)
    state         = Wavefunction(probability_amplitudes=probability_amplitudes, basis=harmonic_oscillator.H.basis)
    
    # ---- Evolution According to Schrodinger Equation ----
    T = 5       # [s]
    dt = 0.0001   # [s]
    num_steps = int(np.round(T / dt))
    
    t_vec = [] # to store time
    populations = [[] for i in range(n_cut)] # to store measurement probabilities
    
    plt.ion()
    fig = plt.figure(figsize=(16, 7))
    
    # ---- Bloch Sphere Subplot ----
    ax_bloch = fig.add_subplot(1, 2, 1, projection='3d')
    
    # ---- Fock Populations Subplot ----
    ax_fock = fig.add_subplot(1, 2, 2) 
    
    for i in range(num_steps):
        t = i * dt
        U = expm((-1j * harmonic_oscillator.H.matrix * t) / hbar)
        state = initial_state.apply(U=U)
        
        probabilities = state.get_probabilities()
        
        print(f" c0: {probabilities[0]}")
        print(f" c1: {probabilities[1]}")
        
        for idx, p in enumerate(probabilities):
            populations[idx].append(p)
        
        t_vec.append(t)
        
        # ---- Bloch Sphere ----
        ax_bloch.cla()
        
        # Wireframe sphere instead of surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(u.size), np.cos(v))
        ax_bloch.plot_wireframe(x, y, z, alpha=0.08, color='gray', linewidth=0.5)
        
        # Draw axis lines
        ax_bloch.plot([-1, 1], [0, 0], [0, 0], color='gray', linewidth=0.5, linestyle='--')
        ax_bloch.plot([0, 0], [-1, 1], [0, 0], color='gray', linewidth=0.5, linestyle='--')
        ax_bloch.plot([0, 0], [0, 0], [-1, 1], color='gray', linewidth=0.5, linestyle='--')
        
        # Label poles
        ax_bloch.text(0, 0, 1.15, "|0⟩", ha='center', fontsize=12, fontweight='bold')
        ax_bloch.text(0, 0, -1.15, "|1⟩", ha='center', fontsize=12, fontweight='bold')
        ax_bloch.text(1.15, 0, 0, "X", ha='center', fontsize=10, color='gray')
        ax_bloch.text(0, 1.15, 0, "Y", ha='center', fontsize=10, color='gray')
    
        
        state_azimuth, state_inclination = get_spherical_coords(alpha=state.probability_amplitudes[0],
                                                                beta=state.probability_amplitudes[1])
        bx, by, bz = get_rectangular_coords(azimuth=state_azimuth, inclination=state_inclination)


        ax_bloch.quiver(0, 0, 0, bx, by, bz, color='red', arrow_length_ratio=0.08, linewidth=2.5)
        ax_bloch.scatter([bx], [by], [bz], color='red', s=40, zorder=5)
        
        ax_bloch.set_xlim([-1.3, 1.3])
        ax_bloch.set_ylim([-1.3, 1.3])
        ax_bloch.set_zlim([-1.3, 1.3])
        ax_bloch.set_box_aspect([1, 1, 1])
        ax_bloch.set_title("Bloch Sphere", fontsize=14, pad=10)
        ax_bloch.set_axis_off()
        ax_bloch.view_init(elev=20, azim=30)
        
        # ---- Fock Populations ----
        ax_fock.cla()
        ax_fock.bar(np.arange(n_cut), probabilities, color='steelblue')
        ax_fock.set_xlim(-0.5, n_cut - 0.5)
        ax_fock.set_ylim(0, 1)
        ax_fock.set_xlabel("Fock State |n⟩", fontsize=12)
        ax_fock.set_ylabel("Probability", fontsize=12)
        ax_fock.set_title(f"Fock Populations (t = {t:.2f} s)", fontsize=14)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
            
    plt.ioff()
    plt.show()
        
    
if __name__=="__main__":
    main()