import numpy as np
import math
import matplotlib.pyplot as plt
class Hilbert:
    def __init__(self, precision: int):
        self.precision = precision
        # Define H matrices
        self.H = [
            np.array([[0, 1], [1, 0]]),
            np.eye(2),
            np.eye(2),
            np.array([[0, -1], [-1, 0]])
        ]
        # Define h vectors
        self.h = [
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([1, 0])
        ]
        # Final F vectors (centers)
        self.F = [
            np.array([1/4, 1/4]),
            np.array([1/4, 3/4]),
            np.array([3/4, 3/4]),
            np.array([3/4, 1/4])
        ]

    def dec_to_quarter(self, number: float):
        q_num = []
        i = 0
        if 0 < number < 1:
            while number != 0 and i < self.precision:
                number *= 4
                digit = math.floor(number)
                q_num.append(digit)
                number -= digit
                i += 1
        elif number == 0:
            q_num.append(0)
        elif number == 1.0:
            q_num.append(1)
        return q_num

    def e_and_d_counters(self, q_num):
        # For each position j, e0j is parity of number of 0s in q_1...q_{j-1}
        # e3j is parity of number of 3s in q_1...q_{j-1}
        e0j = np.zeros(len(q_num), dtype=int)
        e3j = np.zeros(len(q_num), dtype=int)
        for i in range(1, len(q_num)):
            e0j[i] = np.count_nonzero(np.array(q_num[:i]) == 0) % 2
            e3j[i] = np.count_nonzero(np.array(q_num[:i]) == 3) % 2
        return e0j, e3j

    def hilbert_modified(self, q_num):
        n = len(q_num)
        e0j, e3j = self.e_and_d_counters(q_num)
        # Product H_{q1}...H_{qn-1}
        Hprod = np.eye(2)
        for idx in q_num[:-1]:
            Hprod = Hprod @ self.H[idx]
        term1 = (1/2) * Hprod @ self.F[q_num[-1]]

        # Summation term
        sum_j = np.zeros(2)
        for j in range(1, n):
            Hj = (np.linalg.matrix_power(self.H[0], e0j[j]) @
                  np.linalg.matrix_power(self.H[3], e3j[j]) @
                  self.h[q_num[j-1]]
                 )
            sum_j += (1/(2**j)) * Hj
        return term1 + sum_j
    
    def plot_nicer_hilbert_polygon(self, n):
        

        points = []
        for k in range(4 ** n):
            t = k / (4 ** n)
            q = self.dec_to_quarter(t)
            p = self.hilbert_modified(q)
            points.append(p)

        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], '-o', markersize=2)
        plt.axis('equal')
        plt.show()


h2d = Hilbert(6)
h2d.plot_nicer_hilbert_polygon(4)