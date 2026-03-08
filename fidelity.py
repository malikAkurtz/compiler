import numpy as np

def get_leakage(U_proj: np.ndarray):
    return 1 - ( (np.trace(U_proj @ U_proj.conjugate().T)) / 2 )

def get_process_fidelity(U_proj: np.ndarray, U_target: np.ndarray):
    return np.abs( np.trace(U_target.conjugate().T @ U_proj) )**2 / 4

def get_average_gate_fidelity(process_fidelity: float, leakage: float):
    return ( (2 * process_fidelity) + 1 - leakage ) / 3