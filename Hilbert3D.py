import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Hilbert3D:
    
    def __init__(self, precision: int):
        self.precision = precision

    def dec_to_octal(self, number: float):
        q_num = []
        i = 0

        if 0 < number < 1:
            while i < self.precision:
                number *= 8
                digit = math.floor(number)
                q_num.append(digit)
                number -= digit
                i += 1
        elif number == 0.0:
            q_num = [0] * self.precision
        
        elif number == 1.0:
            q_num = [1] + [0] * self.precision-1
            
        return q_num
    
    
    def ThreeD_Hilbert(self, q_num):
        H_all = [
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
        ]

        h_all = [
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 1]),
            np.array([2, 1, 1]),
            np.array([1, 1, 1]),
            np.array([1, 1, 2]),
            np.array([0, 1, 2])
        ]
        
        soucin = np.eye(3)
        s = 0.5 * h_all[q_num[0]]

        for j in range(1, len(q_num)):
            
            soucin = soucin @ H_all[q_num[j-1]]
            s += (1/(2**(j+1))) * (soucin @ h_all[q_num[j]])
            

        return s
    

    def calculate_mainstream_point(self, q_num, n):
        H = [
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
        ]

        F0 = np.array([1/2, 1/2, 1/2])@ H[0]
        F1 = np.array([1/2, 1/2, 1/2])@ H[1]
        F2 = np.array([1/2, 1/2, 1/2])@ H[2]
        F3 = np.array([1/2, 1/2, 1/2])@ H[3]
        F4 = np.array([1/2, 1/2, 1/2])@ H[4]
        F5 = np.array([1/2, 1/2, 1/2])@ H[5]
        F6 = np.array([1/2, 1/2, 1/2])@ H[6]
        F7 = np.array([1/2, 1/2, 1/2])@ H[7]

        F = [F0, F1, F2, F3, F4, F5, F6, F7]
       
        soucin = np.eye(3)
        
        
        for j in (q_num[:-1]):
            
            soucin = soucin @ H[j]
       
        prvniScitanec = (1/2)**(n) * soucin @ F[q_num[-1]]
      
        
        s=self.ThreeD_Hilbert(q_num)
       
        s = s.flatten()
       
        d = s + prvniScitanec
        
        return  d



    def ThreeD_Hilbert_point(self, t):
        
        if t == 1.0:
            return np.array([0.0, 0.0, 1.0])
        elif t > 1.0:
            return np.array([0.0, 0.0, 0.0])
        q = self.dec_to_octal(t)
        point = self.ThreeD_Hilbert(q)
        return point.flatten()


    def hilbert_polygon_point(self, t, n):
    
        N = 2**(3*n)
        
    
        k = int(np.floor(t * N))
       
        p_k = self.ThreeD_Hilbert_point(k / N)
        p_k1 = self.ThreeD_Hilbert_point((k + 1) / N)
     
        point = N * (t-(k/N))*p_k1 - N*(t-((k+1)/N))*p_k

        return point
    
    

# --- Optimalizace ---

    @staticmethod
    def f(x, y, z):
        return ((x - 0.3)**2 + (y - 0.7)**2 + (z - 0.1)**2)

    @staticmethod
    def f1(x, y, z):
        # Schwefel 1.2 - natural domain [-5, 5]^3, global min = 0 at (0, 0, 0)
        s1 = x
        s2 = x + y
        s3 = x + y + z
        return s1**2 + s2**2 + s3**2

    @staticmethod
    def f1_cube(x, y, z):
        # Schwefel 1.2 scaled FROM [-5, 5]^3 TO [0,1]^3, global min at (0.5, 0.5, 0.5)
        x_min, x_max = -5.0, 5.0
        u = x_min + x * (x_max - x_min)
        v = x_min + y * (x_max - x_min)
        w = x_min + z * (x_max - x_min)
        return Hilbert3D.f1(u, v, w)

    @staticmethod
    def f2(x, y, z):
     
        return abs(x-0.1) + abs(y - 0.55) + abs(z - 0.55)

    @staticmethod
    def f2_cube(x, y, z):
        # Levy scaled FROM [-10, 10]^3 TO [0,1]^3, global min at (0.55, 0.55, 0.55)
        x_min, x_max = 0, 1
        u = x_min + x * (x_max - x_min)
        v = x_min + y * (x_max - x_min)
        w = x_min + z * (x_max - x_min)
        return Hilbert3D.f2(u, v, w)

    def evaluate_function(self, x, y, z, whatFunc=0):
        if whatFunc == 0:
            return self.f(x, y, z)
        elif whatFunc == 1:
            return self.f1_cube(x, y, z)
        elif whatFunc == 2:
            return self.f2_cube(x, y, z)
        raise ValueError("whatFunc must be 0, 1 or 2.")

    def F(self, t, n, whatFunc=0):
        x, y, z = self.hilbert_polygon_point(t,n)
        return self.evaluate_function(x, y, z, whatFunc)
   
    # --- Zabudovana python metoda pro hledani minima ---
   
    def find_minimum(self, n, whatFunc=0):
        result = minimize_scalar(lambda t: self.F(t, n, whatFunc), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.hilbert_polygon_point(t_min,n)
        f_min = self.evaluate_function(*h_min, whatFunc=whatFunc)
        return t_min, h_min, f_min

    def differential_evolution_global(self, true_min=None, ftol=1e-6, maxiter=200, whatFunc=0):
        if ftol <= 0:
            raise ValueError("ftol must be positive.")

        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        def objective(coords):
            x, y, z = coords
            return self.evaluate_function(x, y, z, whatFunc)

        generation_count = [0]

        def callback(xk, convergence=0):
            generation_count[0] += 1
            if true_min is not None:
                if abs(objective(xk) - float(true_min)) < float(ftol):
                    print(
                        f"Differential evolution stopped after {generation_count[0]} generations - ftol condition satisfied."
                    )
                    return True
            return False

        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        result = differential_evolution(objective, bounds, maxiter=maxiter, callback=callback)

        x_min, y_min, z_min = result.x
        f_min = float(result.fun)
        nfev = int(result.nfev)
        print(
            f"Differential evolution completed: {generation_count[0]} generations, {nfev} function evaluations, f_min = {f_min}"
        )
        return f_min, x_min, y_min, z_min, generation_count[0], nfev
   
    # --- Optimalizacni algoritmus pro hledani minima ---
    def Holder_algorithm(self, H, r, eps, max_iter, n, true_min=None, ftol=1e-6, stop_condition="eps", I=1, whatFunc=0, return_iterations=False):
        N = 3
        stop_condition = stop_condition.lower()
        if stop_condition not in {"eps", "ftol"}:
            raise ValueError("stop_condition must be either 'eps' or 'ftol'.")

        if stop_condition == "ftol" and true_min is None:
            raise ValueError("true_min must be provided when stop_condition='ftol'.")

        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0.0, n, whatFunc), self.F(1.0, n, whatFunc)]

        # SELECT(2) state variables
        flag = 0
        imin = 0
        side_flag = 0

        actual_iterations = 0
        for iteracni_krok in range(max_iter):
            actual_iterations = iteracni_krok + 1
            # STEP 1: serazeni bodu podle x
            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty
            if H == -1:
                hvalues = []
                for i in range(1, len(xk)):
                    diff = abs(zk[i] - zk[i - 1]) / (abs(xk[i] - xk[i - 1])) ** (1 / N) if abs(xk[i] - xk[i - 1]) > 0 else 0
                    hvalues.append(diff)
                h_value = max(max(hvalues), 1e-8) if hvalues else 1e-8
                h_used = [h_value] * max(1, len(xk) - 1)

            elif H == -2:
                m_values = []
                for i in range(1, len(xk)):
                    diff = abs(zk[i] - zk[i - 1]) / (abs(xk[i] - xk[i - 1])) ** (1 / N) if abs(xk[i] - xk[i - 1]) > 0 else 0
                    m_values.append(diff)

                if not m_values:
                    h_used = [1e-8]
                else:
                    if len(m_values) == 1:
                        lambda_values = [m_values[0]]
                    else:
                        lambda_values = []
                        for i in range(1, len(m_values) - 1):
                            lambda_values.append(max(m_values[i - 1], m_values[i], m_values[i + 1]))
                        lambda_values = [max(m_values[0], m_values[1])] + lambda_values + [max(m_values[-2], m_values[-1])]

                    X_max = max(abs(xk[i] - xk[i - 1]) ** (1 / N) for i in range(1, len(xk)))
                    h_k = max(m_values)
                    gamma_values = []
                    for i in range(1, len(xk)):
                        gamma_values.append(h_k * abs(xk[i] - xk[i - 1]) ** (1 / N) / X_max)

                    xi_param = 1e-8
                    h_values = []
                    for i in range(len(lambda_values)):
                        if i < len(gamma_values):
                            h_values.append(max(lambda_values[i], gamma_values[i], xi_param))
                        else:
                            h_values.append(max(lambda_values[i], xi_param))
                    h_used = h_values
            else:
                h_value = max(H, 1e-8)
                h_used = [h_value] * max(1, len(xk) - 1)

            # STEP 3: vypocet pruseciku a M_i
            Mi = []
            yi = []
            for i in range(1, len(xk)):
                h_i = max(h_used[i - 1], 1e-8)
                y = 0.5 * (xk[i - 1] + xk[i]) - (zk[i] - zk[i - 1]) / (2 * r * h_i * (xk[i] - xk[i - 1]) ** ((1 - N) / N))
                yi.append(y)
                Mi.append(min(zk[i - 1] - r * h_i * abs(y - xk[i - 1]) ** (1 / N), zk[i] - r * h_i * abs(xk[i] - y) ** (1 / N)))

            # STEP 4: vyber intervalu (SELECT(1) nebo SELECT(2))
            if I == 1:
                idx = int(np.argmin(Mi))
            else:
                idx = int(np.argmin(Mi))
                current_imin = int(np.argmin(zk))

                if flag == 1:
                    if len(zk) > 1 and zk[-1] < zk[imin]:
                        imin = len(zk) - 1

                    delta = 1e-5
                    if imin >= 1 and imin < len(xk) - 1:
                        left_size = abs(xk[imin] - xk[imin - 1]) if imin > 0 else 0
                        right_size = abs(xk[imin + 1] - xk[imin]) if imin + 1 < len(xk) else 0

                        if side_flag == 0:
                            if right_size > delta and imin < len(Mi):
                                t_choice = imin
                            elif left_size > delta and imin - 1 < len(Mi):
                                t_choice = imin - 1
                            else:
                                t_choice = int(np.argmin(Mi))
                        else:
                            if left_size > delta and imin - 1 < len(Mi):
                                t_choice = imin - 1
                            elif right_size > delta and imin < len(Mi):
                                t_choice = imin
                            else:
                                t_choice = int(np.argmin(Mi))

                        side_flag = 1 - side_flag

                        if 0 <= t_choice < len(Mi):
                            interval_size = abs(xk[t_choice + 1] - xk[t_choice])
                            if interval_size > delta:
                                idx = t_choice

                    flag = 0
                else:
                    flag = 1
                    idx = int(np.argmin(Mi))
                    imin = current_imin

            y_star = yi[idx]

            # STEP 5: zastavovaci podminka
            if stop_condition == "ftol":
                current_best = min(zk)
                if abs(current_best - float(true_min)) < float(ftol):
                    print(f"Holder algorithm stopped after {iteracni_krok + 1} iterations - ftol condition satisfied.")
                    break
            else:
                if abs(xk[idx + 1] - xk[idx]) ** (1 / N) < eps:
                    print(f"Holder algorithm stopped after {iteracni_krok + 1} iterations - smallest interval below threshold.")
                    break

            xk.append(y_star)
            zk.append(self.F(y_star, n, whatFunc))

        min_idx = int(np.argmin(zk))
        t_min = xk[min_idx]
        x_min, y_min, z_min = self.hilbert_polygon_point(t_min, n)
        f_min = self.evaluate_function(x_min, y_min, z_min, whatFunc)

        if return_iterations:
            return t_min, f_min, x_min, y_min, z_min, actual_iterations
        return t_min, f_min, x_min, y_min, z_min