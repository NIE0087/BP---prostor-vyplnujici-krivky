import math
import numpy as np
from scipy.optimize import minimize_scalar


class Hilbert2D:
    def __init__(self, precision: int):
        self.precision = precision

    # --- Konverze ---
    def dec_to_octal(self, number: float):
        q_num = []
        i = 0
        if 0 < number < 1:
            while number != 0 and i < self.precision:
                number *= 8
                digit = math.floor(number)
                q_num.append(digit)
                number -= digit
                i += 1
        return q_num

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
        elif number == 1:
            q_num.append(1)
        elif number == 0:
            q_num.append(0)
        return q_num

    # --- Hilbert 2D ---
    def ej_and_dj_counter(self, q_num):
        e0j_counted = np.zeros(len(q_num))
        e3j_counted = np.zeros(len(q_num))
        dj_counted = np.zeros(len(q_num))

        for i in range(1, len(q_num)):
            for j in range(i):
                if q_num[j] == 3:
                    e3j_counted[i] += 1
                elif q_num[j] == 0:
                    e0j_counted[i] += 1
        e3j_counted %= 2
        e0j_counted %= 2
        dj_counted = (e3j_counted + e0j_counted) % 2
        return e0j_counted, dj_counted

    def calculate_point(self, e0j_counted, dj_counted, q_num):
        s = np.zeros((2, 1))
        for i in range(1, len(q_num) + 1):
            s += (1/(2**i)) * ((-1)**e0j_counted[i-1]) * (
                np.sign(q_num[i-1]) *
                np.array([[(1-dj_counted[i-1])*q_num[i-1]-1],
                          [1-dj_counted[i-1]*q_num[i-1]]])
            )
        return s

    def hilbert_point(self, t):
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_point(e0, dj, q)
        return point.flatten()

    # --- Optimalizace ---
    @staticmethod
    def f(x, y):
        return (x - 0.5)**2 + (y - 0.5)**2

    def F(self, t):
        x, y = self.hilbert_point(t)
        return self.f(x, y)

    def find_minimum(self):
        result = minimize_scalar(lambda t: self.F(t), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.hilbert_point(t_min)
        f_min = self.f(*h_min)
        return t_min, h_min, f_min


class Hilbert3D:
    def ThreeD_Hilbert(self, q_num):
        H_all = [
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
        ]

        h_all = [
            np.array([0, 0, 0]),
            np.array([0, 1, 1]),
            np.array([1, 1, 0]),
            np.array([1, 1, 1]),
            np.array([2, 1, 1]),
            np.array([1, 1, 1]),
            np.array([1, 1, 2]),
            np.array([0, 1, 2])
        ]

        soucin = np.eye(3)
        s = h_all[q_num[0]] * 0.5

        for j in range(1, len(q_num)):
            for k in range(j):
                soucin = soucin @ H_all[q_num[k]]
            s += (1/(2**(j+1))) * (soucin @ h_all[q_num[j]])
            soucin = np.eye(3)

        return s
