import numpy as np
# np.set_printoptions(precision=2, suppress=True)

from scipy.linalg import expm
import matplotlib.pyplot as plt

from System import System
from HarmonicOscillator import HarmonicOscillator
from SFQDriver import SFQDriver
from Wavefunction import Wavefunction
from utils import *
from constants import *
from fidelity import *
from itertools import combinations

def problem_2():
    A = np.array([
        [-6,-8,-4,-3,0,0,0,0,0],
        [1,1,0,0,1,0,0,0,48],
        [0,0,1,1,0,1,0,0,60],
        [1,0,1,0,0,0,1,0,36],
        [0,1,0,4,0,0,0,1,72]
    ], dtype=float)
    
    pivot_row = A[1]
    
    A[0] = A[0] + (8 * pivot_row)
    A[4] = A[4] + (-1 * pivot_row)
    
    A[2] = A[2] + (-1 * A[3])
    A[0] = A[0] + (4 * A[3])
    
    A[4] = A[4] / 4
    A[2] = A[2] + (-1 * A[4])
    A[0] = A[0] + (3 * A[4])
    
    print(A)
    
def problem_4():
    A = np.array([
        [1, 0, 0, 0],   # C1: x1 >= 0
        [0, 1, 0, 0],   # C2: x2 >= 0
        [0, 0, 1, 0],   # C3: x3 >= 0
        [0, 0, 0, 1],   # C4: x4 >= 0
        [1, 2, 3, 4],   # C5: x1+2x2+3x3+4x4 <= 12
        [3, 1, 0, 2],   # C6: 3x1+x2+2x4 <= 6
    ])

    b = np.array([0, 0, 0, 0, 12, 6])

    vertices = []
    for combo in combinations(range(6), 4):
        row_indices = list(combo)
        
        active_rows = A[row_indices]
        active_rhs  = b[row_indices]
        
        if np.allclose(np.abs(np.linalg.det(active_rows)), 0.0):
            continue  # linearly dependent
        else:
            x = np.linalg.solve(active_rows, active_rhs)
            
            # check all 6 constraints
            if np.all(x >= -1e-10) and np.all(A[4:] @ x <= b[4:] + 1e-10):
                vertices.append(np.round(x, 6))
                
    for v in vertices:
        print(v)
        
        
def problem_5():
    A_B_inv_b = np.array([
        [2],
        [3],
        [1]
    ], dtype=float)
    
    A_B_inv = np.array([
        [1/2, 1/5, -1],
        [-1, 0, 1/2],
        [5, -3/10, 2],
    ], dtype=float)
    
    A_B = np.linalg.inv(A_B_inv)
    
    b = A_B @ A_B_inv_b
    
    # from strong duality
    y = np.array([
        [2],
        [1/10],
        [2]
    ], dtype=float)
    
    b_T_y = b.T @ y
    
    theta = b_T_y
    
    print(theta)
    
if __name__=="__main__":
    
    problem_5()