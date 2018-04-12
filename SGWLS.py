import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import colorspacious as cs

def extractColumnNeighborhood(A, k, r):
    Nb = A[:, k-r:k+r+1]
    for i in range(1, A.shape[0], 2):
        Nb[i, :] = np.flip(Nb[i, :], axis=0)
    return Nb.flatten(order="C")
    
    
