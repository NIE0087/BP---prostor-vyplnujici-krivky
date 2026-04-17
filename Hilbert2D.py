import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, OptimizeResult, minimize
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
class Hilbert2D:

    def __init__(self, precision: int):
        """Nastaví přesnost kvartérního rozvoje pro parametr t."""
        self.precision = precision




#################################################################
# ---------- SESTROJENÍ ITERACÍ HILBERTOVY KŘIVKY --------------#
#################################################################
    
    # --- Konverze ---
    def dec_to_quarter(self, number: float):
        """Převede číslo z intervalu [0,1] na kvartérní cifry délky precision."""
        q_num = []
        i = 0
        if 0 < number < 1:
            while  i < self.precision:
                number *= 4
                digit = math.floor(number)
                q_num.append(digit)
                number -= digit
                i += 1
        elif number == 0.0:
            q_num = [0] * self.precision
        
        elif number == 1.0:
            q_num =[1] + [0] * self.precision
        
        return q_num



    # --- Hilbert 2D vzorec ---
    def ej_and_dj_counter(self, q_num):
        """Spočítá pomocné parity e0j a dj pro konstrukci bodu křivky."""
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
        """Vypočítá bod Hilbertovy křivky z kvartérních číslic."""
        s = np.zeros((2, 1))
        for i in range(1, len(q_num) + 1):
            s += (1/(2**i)) * ((-1)**e0j_counted[i-1]) * (
                np.sign(q_num[i-1]) *
                np.array([[(1-dj_counted[i-1])*q_num[i-1]-1],
                          [1-dj_counted[i-1]*q_num[i-1]]])
            )
        return s
    


    def hilbert_point(self, t):
        """Vrátí bod ideální 2D Hilbertovy křivky pro parametr t."""
        if t == 1.0:
            return np.array([1.0, 0.0]) 
        
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_point(e0, dj, q)
        return point.flatten()
    

    def hilbert_polygon_point(self, t, n):
        """Vrátí bod polygonální aproximace Hilbertovy křivky řádu n."""
    
        N = 2**(2*n)
    
    
        k = int(np.floor(t * N))
   

        p_k = self.hilbert_point(k / N)
        p_k1 = self.hilbert_point((k + 1) / N)
  
        point = N * (t-(k/N))*p_k1 - N*(t-((k+1)/N))*p_k

        return point


    

#################################################################
# ------------ ZKUŠEBNÍ FUNKCE PRO OPTIMALIZACI-----------------#
#################################################################



    @staticmethod
    # --- Random function ---
    def f(x, y):
        """Jednoduchá testovací funkce se známým minimem."""
        return ((x - 0.3)**2 + (y - 0.7)**2)**(1/2) + 1
    
    
     
    # --- Branin function ---
    @staticmethod
    def f1(x, y):
        """Braninova testovací funkce pro globální optimalizaci."""
        
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a * (y - b*x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s
        
    

    # --- Matyas function ---
    @staticmethod
    def f2(x, y):
        """Matyasova testovací funkce pro optimalizaci ve 2D."""
        return 0.26*(x**2 + y**2) - 0.48*y*x
    
     
#################################################################
# --------------------- MAPOVÁNÍ FUNKCÍ-------------------------#
#################################################################


    @staticmethod
    def f1_square(x, y):
        """Přemapuje [0,1]^2 na oblast Braninovy funkce a vyhodnotí ji."""
        x_min=-5
        x_max=10
        y_min=0
        y_max=15
       
        u = x_min + x * (x_max - x_min)
        v = y_min + y * (y_max - y_min)


        return Hilbert2D.f1(u,v)
        
    

    # --- Matyas function ---
    @staticmethod
    def f2_square(x, y):
        """Přemapuje [0,1]^2 na oblast Matyasovy funkce a vyhodnotí ji."""
        x_min=-10
        x_max=10
        y_min=-10
        y_max=10
       
        u = x_min + x * (x_max - x_min)
        v = y_min + y * (y_max - y_min)


        return Hilbert2D.f2(u,v)
    
    
    #---- Složená funkce -----
    def F(self, t, n, whatFunc):
        """Složená 1D funkce: Hilbertovo mapování + výběr testovací funkce."""
        x, y = self.hilbert_polygon_point(t,n)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1_square(x,y)
        else:
            return self.f2_square(x,y)


    def map_to_area(self, t,n, x_min, x_max, y_min, y_max):
        """Přemapuje bod křivky z [0,1]^2 do zadaného obdélníku."""
        point = self.hilbert_polygon_point(t,n)
        px, py = point
        new_x = x_min + (x_max - x_min) * px
        new_y = y_min + (y_max - y_min) * py
        return np.array([new_x, new_y])


    def F_mapped(self, t, n, x_min, x_max, y_min, y_max, whatFunc):
        """Vyhodnotí zvolenou funkci po mapování do zadané oblasti."""
        x, y = self.map_to_area(t, n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1(x,y)
        else:
            return self.f2(x,y)
    

#################################################################
# ------------ ALGORITMY HLEDAJÍCÍ MINIMA ----------------------#
#################################################################



    def differential_evolution_mapped(self, x_min, x_max, y_min, y_max, whatFunc, true_min=None, ftol=1e-6, maxiter=200):
        """Najde minimum pomocí differential evolution v mapované oblasti."""
        
        def objective(coords):
            x, y = coords
            if whatFunc == 0:
                return self.f(x, y)
            elif whatFunc == 1:
                return self.f1(x, y)
            else:
                return self.f2(x, y)
        
   
        iteration_count = [0]
        def callback(xk, convergence=0):
            iteration_count[0] += 1
            if true_min is not None:
                current_f = objective(xk)
                if np.abs(current_f - true_min) < ftol:
                    print(f"Differential evolution stopped after {iteration_count[0]} iterations - desired accuracy achieved.")
                    return True 
            return False
        
        bounds = [(x_min, x_max), (y_min, y_max)]
        
        result = differential_evolution(objective, bounds, maxiter=maxiter, callback=callback)
        
        x_min_de, y_min_de = result.x
        f_min = result.fun
        
     
        print(f"Differential evolution completed: {iteration_count[0]} generations, {result.nfev} function evaluations, f_min = {f_min}")
        
        return f_min, x_min_de, y_min_de, iteration_count[0], result.nfev





    def Holder_algorithm_mapped(self,H,I, r,eps,max_iter,n, whatFunc, true_min, ftol, stop_condition="eps"):
        """Spustí Holderův 1D algoritmus nad Hilbertovsky mapovanou funkcí."""
        N = 2                      
        stop_condition = stop_condition.lower()
        if stop_condition not in {"eps", "ftol"}:
            raise ValueError("stop_condition must be either 'eps' or 'ftol'.")

        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0.0,n, whatFunc), self.F(1.0,n, whatFunc)]
        k = 2
        usedH_arr = []  
        
        # SELECT(2) konstanty
        flag = 0  
        imin = 0  
        side_flag = 0  
        z_new = None  # hodnota naposledy přidaného bodu (před sortem)

        for iteracni_krok in range(max_iter):
            
            # STEP 1: serazeni bodu podle hodnoty

            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty
            if H == -1:
                h_used, h_value = self.HOLDER_CONST_1(xk, zk, N)
            elif H == -2:
                h_used, h_value = self.HOLDER_CONST_2(xk, zk, N)
            else:
                h_value = H
                h_used = [h_value] * (len(xk) - 1)

            usedH_arr.append(h_value)
          
            # STEP 3: vypocet pruseciku a M_i
            Mi = []
            yi = []
            
            for i in range(1, len(xk)):
                h_i = max(h_used[i-1], 1e-8)
                y = 0.5*(xk[i-1] + xk[i]) - (zk[i] - zk[i-1])/(2*r*h_i*(xk[i]-xk[i-1])**((1-N)/N))
                yi.append(y)
                # Vypocet M_i 
                Mi.append(min(zk[i-1] - r*h_i * abs(y - xk[i-1])**(1/N), zk[i] - r*h_i * abs(xk[i] - y)**(1/N)))
            
            # STEP 4: vyber intervalu
            if I == 1:
                idx, y_star = self.SELECT_1(Mi, yi)
            else:
                idx, y_star, flag, imin, side_flag = self.SELECT_2(Mi, yi, xk, zk, flag, imin, eps, side_flag, z_new)
            
            # STEP 5: zastavovaci podminka
            
            min_idx = np.argmin(zk)
            t_current = xk[min_idx]
            x_current, y_current = self.hilbert_polygon_point(t_current,n)
            
            if whatFunc == 0:
                f_current = self.f(x_current, y_current)
            elif whatFunc == 1:
                f_current = self.f1_square(x_current, y_current)
            else:
                f_current = self.f2_square(x_current, y_current)
            
            if stop_condition == "ftol":
                if abs(f_current - true_min) < ftol:
                    print(f"Algorithm stopped after {iteracni_krok + 1} iterations - ftol condition satisfied.")
                    break
            else:
                if len(xk) > 1:
                    min_interval = min(
                        abs(xk[i] - xk[i - 1]) ** (1 / N)
                        for i in range(1, len(xk))
                    )
                    if min_interval < eps:
                        print(f"Algorithm stopped after {iteracni_krok + 1} iterations - smallest interval below threshold.")
                        break
           

            xk.append(y_star)
            z_new = self.F(y_star, n, whatFunc)
            zk.append(z_new)
            k += 1
        
        min_idx = np.argmin(zk)
        t_min = xk[min_idx]               # parametr t na Hilbertově křivce
        x_min_mapped, y_min_mapped = self.hilbert_polygon_point(t_min,n)  # souřadnice v R^2
        if whatFunc==0:
            f_min = self.f(x_min_mapped, y_min_mapped)      # hodnota funkce f(x,y)
        elif whatFunc==1:
            f_min = self.f1_square(x_min_mapped, y_min_mapped)
        else:
            f_min = self.f2_square(x_min_mapped,y_min_mapped)
    
        return t_min, f_min, x_min_mapped, y_min_mapped, usedH_arr


    def HOLDER_CONST_1(self, xk, zk, N):
        """Spočítá globální odhad Holderovy konstanty pro všechny intervaly."""
        hvalues = []
        for i in range(1, len(xk)):
            diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1]) > 0 else 0
            hvalues.append(diff)

        h_hat = max(hvalues)
        h_value = max(h_hat, 1e-8)
        h_used = [h_value] * len(hvalues)
        return h_used, h_value


    def HOLDER_CONST_2(self, xk, zk, N):
        """Spočítá lokální odhady Holderovy konstanty podle okolních intervalů."""
        m_values = []
        for i in range(1, len(xk)):
            diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1]) > 0 else 0
            m_values.append(diff)

        lambda_values = []
        if len(xk) == 2:
            lambda_values.append(m_values[0])
        else:
            for i in range(1, len(m_values)-1):
                lambda_i = max(m_values[i-1], m_values[i], m_values[i+1])
                lambda_values.append(lambda_i)

            lambda_2 = max(m_values[0], m_values[1])
            lambda_k = max(m_values[-2], m_values[-1])
            lambda_values = [lambda_2] + lambda_values + [lambda_k]

        gamma_values = []
        X_max = max([abs(xk[i] - xk[i-1])**(1/N) for i in range(1, len(xk))])
        h_k = max(m_values)

        for i in range(1, len(xk)):
            gamma_i = h_k * abs(xk[i] - xk[i-1])**(1/N) / X_max
            gamma_values.append(gamma_i)

        xi_param = 1e-8
        h_values = []
        for i in range(len(lambda_values)):
            if i < len(gamma_values):
                h_i = max(lambda_values[i], gamma_values[i], xi_param)
            else:
                h_i = max(lambda_values[i], xi_param)
            h_values.append(h_i)

        h_value = max(h_values)
        return h_values, h_value


    def SELECT_1(self, Mi, yi):
        """Vybere interval s nejmenší hodnotou minorantu."""
        idx = np.argmin(Mi)
        y_star = yi[idx]
        return idx, y_star

    def SELECT_2(self, Mi, yi, xk, zk, flag, imin, eps, side_flag, z_new=None):
        """Rozšířený výběr intervalu se střídáním stran kolem nejlepšího bodu."""
        idx = np.argmin(Mi)
        delta = eps*10

        if flag == 1:
            # Aktualizuj imin pokud nově přidaný bod je lepší (g(x^k) < g(x_imin))
            # z_new je hodnota naposledy přidaného bodu 
            if z_new is not None and z_new < zk[imin]:
                imin = np.argmin(zk)

            # Lokální zlepšení: střídej strany kolem imin
            if imin >= 1 and imin < len(xk) - 1:
                left_size = abs(xk[imin] - xk[imin - 1])
                right_size = abs(xk[imin + 1] - xk[imin])

                if side_flag == 0:
                    if right_size > delta and imin < len(Mi):
                        t_choice = imin
                    elif left_size > delta and imin - 1 < len(Mi):
                        t_choice = imin - 1
                    else:
                        t_choice = np.argmin(Mi)
                else:
                    if left_size > delta and imin - 1 < len(Mi):
                        t_choice = imin - 1
                    elif right_size > delta and imin < len(Mi):
                        t_choice = imin
                    else:
                        t_choice = np.argmin(Mi)

                side_flag = 1 - side_flag

                if 0 <= t_choice < len(Mi):
                    interval_size = abs(xk[t_choice + 1] - xk[t_choice])
                    if interval_size > delta:
                        idx = t_choice

            flag = 0
        else:
            # flag == 0: globální výběr jako SELECT(1)
            idx = np.argmin(Mi)
            imin = np.argmin(zk)
            flag = 1

        y_star = yi[idx]
        return idx, y_star, flag, imin, side_flag