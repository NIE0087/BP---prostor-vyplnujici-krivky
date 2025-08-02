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
   
    # --- Zabudovana python metoda pro hledani minima ---
   
    def find_minimum(self):
        result = minimize_scalar(lambda t: self.F(t), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.hilbert_point(t_min)
        f_min = self.f(*h_min)
        return t_min, h_min, f_min
   
    # --- Optimalizacni algoritmus pro hledani minima ---
    def Holder_algorithm(self,H,r,eps,max_iter):
        N = 2                      
        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0.0), self.F(1.0)]
        k = 2

        for iteracni_krok in range(max_iter):
            
            # STEP 1: serazeni bodu podle hodnoty

            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty
            hvalues = []
            #for i in range(1, len(xk)):
            #    diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
            #    hvalues.append(diff)
            h_hat = max(hvalues) if hvalues else H
            h_used = max([h_hat, 1e-8])   
    
            # STEP 3: vypocet pruseciku a M_i
            Mi = []
            yi = []
            for i in range(1, len(xk)):
            
                y = 0.5*(xk[i-1] + xk[i]) - (zk[i] - zk[i-1])/(2*r*h_used*(xk[i]-xk[i-1])**((1-N)/N))
                yi.append(y)
                # Vypocet M_i 
                Mi.append(min(zk[i-1] - r*h_used * abs(y - xk[i-1])**(1/N), zk[i] - r*h_used * abs(xk[i] - y)**(1/N)))
            
            # STEP 4: vyber intervalu 
        
            idx = np.argmin(Mi)
            y_star = yi[idx]
            
            # STEP 5: zastavovaci podminka
            if abs(xk[idx+1] - xk[idx])**(1/N) < eps:
                break

           
            xk.append(y_star)
            zk.append(self.F(y_star))
            k += 1
        
        min_idx = np.argmin(zk)
        t_min = xk[min_idx]               # parametr t na Hilbertově křivce
        x_min, y_min = self.hilbert_point(t_min)  # souřadnice v R^2
        f_min = self.f(x_min, y_min)      # hodnota funkce f(x,y)

        return t_min, f_min, x_min, y_min


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
