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

def main():
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], # obj
        [-1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0], # s1
        [1, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0], # s2
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 3], # w
        [0, 1, 1, 0, -1, 0, 0, 0, 1, 0, 3], # a1
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 5] # a2
    ])
    print("Initial Tableu: ")
    print(A)
    
    A[3] = A[3] + (-1 * A[2])

    A[0] = A[0] - A[4]
    A[0] = A[0] - A[5]
    
    print("After initial preparation: ")
    print(A)
    
    A[0] = A[0] + A[4]
    A[1] = A[1] + A[4]
    
    print("After first pivot: ")
    print(A)
    
    A[2] = A[2] + A[3]
    A[0] = A[0] + A[3]
    A[5] = A[5] - A[3]
    
    print("After second pivot: ")
    print(A)
    
    A[0] = A[0] + A[5]
    A[1] = A[1] + A[5]
    A[3] = A[3] + A[5]
    
    print("After third pivot: ")
    print(A)
    
    print("Done Phase 1")
    
    A = np.delete(A, 8, axis=1)  # remove a1 column
    A = np.delete(A, 8, axis=1)  # remove a2 column (now shifted)
    
    A[0] = np.array([3, 5, 1, 1, -2, 2, 4, 0, 0])
    
    print("After getting rid of artifical obj row and cols: ")
    print(A)
    
    A[0] = A[0] + (-3 * A[5])
    A[0] = A[0] + (-5 * A[4])
    A[0] = A[0] + (-1 * A[3])
    A[0] = A[0] + (-2 * A[1])
    A[0] = A[0] + (-4 * A[2])
    
    print("After zeroing obj row corresponding to basic vars: ")
    print(A)
    
    A[5] = A[5] + A[4]
    A[0] = A[0] + A[4]
    
    print("After first pivot: ")
    print(A)
    
    A[0] = A[0] + (2*A[3])
    A[4] = A[4] + (A[3])
    
    print("After second pivot: ")
    print(A)
    
    # A = np.insert(A, 0, np.array([3, 5, 1, 1, -2, 2, 4, 0]), axis=0)
    
if __name__=="__main__":
    
    main()