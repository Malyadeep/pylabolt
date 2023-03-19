import numpy as np


def get_coeff(lat):
    if lat == 'D2Q9':
        dr = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]]
        ln = [0, 1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        wt = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
        return dr, ln, wt
    elif lat == 'D1Q3':
        dr = [[0, 0], [1, 0], [-1, 0]]
        ln = [0, np.sqrt(2), np.sqrt(2)]
        wt = [2/3, 1/6, 1/6]
        return dr, ln, wt
    else:
        print("Invalid lattice")
