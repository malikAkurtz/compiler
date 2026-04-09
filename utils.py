import numpy as np

from PauliMatrices import PAULI_MATRICES
from Operator import Operator

def get_spherical_coords(alpha, beta):
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    alpha = alpha / norm
    beta = beta / norm
    
    phi = np.arctan2(alpha.imag, alpha.real)
    tau = np.arctan2(beta.imag, beta.real)
    
    azimuth = phi - tau
    
    inclination = 2 * np.arccos(np.clip(np.abs(alpha), -1, 1))
    
    return azimuth, inclination
    
    
def get_rectangular_coords(azimuth, inclination):
    x = np.sin(inclination) * np.cos(azimuth)
    y = np.sin(inclination) * np.sin(azimuth)
    z = np.cos(inclination)
    
    return x, y, z

def create_upper_lower(value: float, dim: int):
    matrix = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            if (i + 1) == j:
                matrix[i][j] = value
            elif (i - 1) == j:
                matrix[i][j] = value
                
    return matrix

def extract_relative_phase(U: np.ndarray):
    if len(U) != 2:
        print(f"Only works for 2x2 unitaries")
        return
    
    alpha = U[0][0]
    beta = U[1][1]
    
    phi = np.angle(beta) - np.angle(alpha)
    
    return phi

def vector_change_basis(transformation_matrix: np.ndarray, vector: np.ndarray):
    return transformation_matrix.conj().T @ vector

def matrix_change_basis(transformation_matrix: np.ndarray, matrix: np.ndarray):
    return transformation_matrix.conj().T @ matrix @ transformation_matrix

def fock_to_phase(self, fock_coefs, phases: np.ndarray):
    N = len(fock_coefs)
    K = len(phases)
    
    linear_map = np.zeros((K, N), dtype=complex)
    
    for k in range(K):
        phase = phases[k]
        for n in range(N):
            linear_map[k][n] = np.exp(-1j * n * phase) / np.sqrt(2*np.pi)
            
    return vector_change_basis(transformation_matrix=linear_map, vector=fock_coefs)
    
def to_ket(coefs: np.ndarray):
    return coefs.reshape(-1, 1)

def to_bra(coefs: np.array):
    return np.array([c.conj() for c in coefs])

def get_pauli_coefs(U: np.ndarray, basis: str):
        if U[basis].shape != (2, 2):
            raise Exception("Matrix is Not (2 x 2)")

        coefs = np.zeros(len(PAULI_MATRICES), dtype=complex)
    
        for idx, matrix in enumerate(PAULI_MATRICES):
            coefs[idx] = np.trace(matrix @ U[basis]) / 2
            
        return coefs

def get_RY_target(theta_target: float):
    RY_TARGET = Operator(
        basis_to_matrix={"energy": np.array([
                [np.cos(theta_target / 2), -np.sin(theta_target / 2)],
                [np.sin(theta_target / 2), np.cos(theta_target / 2)]
            ])}
    )
    return RY_TARGET

def get_RX_target(theta_target: float):
    RX_TARGET = Operator(
        basis_to_matrix={"energy": np.array([
                [np.cos(theta_target / 2), -1j * np.sin(theta_target / 2)],
                [-1j * np.sin(theta_target / 2), np.cos(theta_target / 2)]
            ])}
    )
    return RX_TARGET

