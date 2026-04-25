import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Physical constants
q      = 1.6e-19
h_bar  = 1.05e-34
factor = h_bar / (2 * q)

# Qubit parameters
omega_q = 2 * np.pi * 5e9
alpha   = -2 * np.pi * 250e6
T_q     = 2 * np.pi / omega_q      # qubit period, ~200 ps
Tc      = T_q / 4                   # clock period (4x clock), ~50 ps

# Kick angle (set by Cc/C and circuit params — use the value that matches your transmon sim)
theta = 0.018  # radians per kick, matching CC = 0.003*C

# Target gate
theta_target = np.pi   # Ry(pi) = X gate

# Number of train kicks
N_train = int(round(theta_target / theta))
print(f"Train kicks: {N_train}")

# ---- DRAG ramp construction ----
# Number of on-ramp qubit cycles (Shillito: 4-5 cycles optimal)
n_ramp_cycles = 4

# For each qubit cycle in the ramp, we choose from {0000, 1000, 0100, 1100}
# 0000 = identity, 1000 = Ry(θ), 0100 = Rx(-θ), 1100 = Rx(-θ)Ry(θ)
#
# Optimal 4-cycle on-ramp for Ry(π) at 4x clock (from exhaustive search):
# Each entry is a 4-bit pattern for one qubit cycle
on_ramp  = ['0100', '1100', '0100', '1100']
off_ramp = ['0010', '1010', '0010', '1010']  # Y-symmetric, X-antisymmetric

# Build the full binary sequence
full_sequence = []
for pattern in on_ramp:
    full_sequence.extend([int(b) for b in pattern])

for _ in range(N_train):
    full_sequence.extend([1, 0, 0, 0])  # resonant train: 1000

for pattern in off_ramp:
    full_sequence.extend([int(b) for b in pattern])

# Convert binary sequence to time-voltage pairs
# Each bit corresponds to one clock period Tc
# A '1' means a kick (delta-function SFQ pulse), '0' means free evolution
N_total = len(full_sequence)
T_total = N_total * Tc

print(f"Total clock slots: {N_total}")
print(f"Total gate time: {T_total*1e9:.2f} ns")
print(f"Ramp cycles: {n_ramp_cycles}")

# Build a high-resolution time grid
# Each SFQ pulse is modeled as a narrow Gaussian (width ~ 1 ps)
sigma_pulse = 1e-12  # 1 ps pulse width
dt_grid = sigma_pulse / 5  # resolve each pulse with ~5 points
t_grid = np.arange(0, T_total, dt_grid)

V_grid = np.zeros_like(t_grid)

Phi0 = 2 * np.pi * factor

for i, bit in enumerate(full_sequence):
    if bit == 1:
        t_kick = (i + 0.5) * Tc  # center of clock slot
        # Gaussian approximation to delta: area = Phi_0
        amplitude = Phi0 / (sigma_pulse * np.sqrt(2 * np.pi))
        V_grid += amplitude * np.exp(-0.5 * ((t_grid - t_kick) / sigma_pulse) ** 2)

# Verify total area
area = np.trapezoid(V_grid, t_grid)
n_kicks = sum(full_sequence)
print(f"Total kicks: {n_kicks}")
print(f"Total area: {area:.4e} V·s")
print(f"Expected ({n_kicks} × Φ₀): {n_kicks * Phi0:.4e} V·s")
print(f"Ratio: {area / (n_kicks * Phi0):.6f}")

# Downsample to manageable CSV
n_out = 200_000
t_out = np.linspace(t_grid[0], t_grid[-1], n_out)
V_out = np.interp(t_out, t_grid, V_grid)

out_path = os.path.join(os.path.dirname(__file__), "sfq_V_lookup.csv")
header = "time_s,voltage_V,current_A"
I_dummy = np.zeros(n_out)
np.savetxt(out_path, np.column_stack([t_out, V_out, I_dummy]),
           delimiter=",", header=header, comments="", fmt="%.6e")

print(f"Written: {out_path} ({n_out} rows)")