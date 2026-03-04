import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


from constants import hbar
from System import HarmonicOscillator
from Wavefunction import Wavefunction

def main():
    # ---- Hyper-parameters for Harmonic Oscillator ----
    C     = 100     # [fF]
    L     = 10      # [nH]
    n_cut = 5      # Number of Fock states to truncate to
    
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
    probability_amplitudes = n_cut * [0]
    probability_amplitudes[0] = 1 / np.sqrt(2) 
    probability_amplitudes[1] = 1 / np.sqrt(2) 
    probability_amplitudes = np.array(probability_amplitudes)
    
    state = Wavefunction(probability_amplitudes=probability_amplitudes, basis=harmonic_oscillator.H.basis)
    
    # ---- Evolution According to Schrodinger Equation ----
    T = 5       # [s]
    dt = 0.01   # [s]
    num_steps = int(np.round(T / dt))
    
    t_vec = [] # to store time
    populations = [[] for i in range(n_cut)] # to store measurement probabilities
    
    plt.ion()
    
    fig, ax = plt.subplots()
    
    # Line for P0
    line_state, = ax.plot([], [])
    
    plt.xlim(0, n_cut)
    plt.ylim(0, 1)
    
    for i in range(num_steps):
        t = i * dt
        U = expm((-1j * harmonic_oscillator.H.matrix * t) / hbar)
        state.apply(U=U)
        
        probabilities = state.get_probabilities()
        
        for idx, p in enumerate(probabilities):
            print(idx)
            print(p)
            print(populations)
            populations[idx].append(p)
        
        # Update live plot
        line_state.set_data(np.arange(n_cut), probabilities)
        fig.canvas.draw()
        fig.canvas.flush_events()
            
        t_vec.append(t)
    
    plt.ioff()
    plt.show()
        
    
if __name__=="__main__":
    main()