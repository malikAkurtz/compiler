import numpy as np

from Wavefunction import Wavefunction

def get_spherical_coords(alpha, beta):    
    phi = np.arctan2(alpha.imag, alpha.real)
    tau = np.arctan2(beta.imag, beta.real)
    
    azimuth = phi - tau
    
    inclination = 2 * np.arccos(np.abs(alpha))
    
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
    
        
        
    