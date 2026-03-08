import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, OptimizeResult, minimize
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
try:
    import nlopt
except ImportError:
    nlopt = None


class Hilbert2D:

    def __init__(self, precision: int):
        self.precision = precision




#################################################################
# ---------- SESTROJENÍ ITERACÍ HILBERTOVY KŘIVKY --------------#
#################################################################
    
    # --- Konverze ---
    def dec_to_quarter(self, number: float):
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
        if t == 1.0:
            return np.array([1.0, 0.0]) 
        
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_point(e0, dj, q)
        return point.flatten()
    

    def hilbert_polygon_point(self, t, n):
    
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
        return ((x - 0.3)**2 + (y - 0.7)**2)**(1/2) + 1
    
    
    # --- Booth function --- 
    # --- Three-hump camel function ---
    # --- Easom function ---
    @staticmethod
    def f1(x, y):
        #return (x + 2*y - 7)**2 + (y + 2*x - 5)**2
        return np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
        #return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    

    # --- Matyas function ---
    @staticmethod
    def f2(x, y):
        return 0.26*(x**2 + y**2) - 0.48*y*x
    
     
#################################################################
# --------------------- MAPOVÁNÍ FUNKCÍ-------------------------#
#################################################################


    @staticmethod
    def f1_square(x, y):
        x_min=-1.5
        x_max=4.5
        y_min=-3
        y_max=4.5
       
        u = x_min + x * (x_max - x_min)
        v = y_min + y * (y_max - y_min)


        return Hilbert2D.f1(u,v)
        #return 2*x**2 - 1.05*x**4 + (x**6)/6 + y*x + y**2
        #return -math.cos(x)*math.cos(y)*math.exp(-((x-math.pi)**2 + (y-math.pi)**2))
    

    # --- Matyas function ---
    @staticmethod
    def f2_square(x, y):
        x_min=-10
        x_max=10
        y_min=-10
        y_max=10
       
        u = x_min + x * (x_max - x_min)
        v = y_min + y * (y_max - y_min)


        return Hilbert2D.f2(u,v)
    
    
    #---- Složená funkce -----
    def F(self, t, n, whatFunc):
        x, y = self.hilbert_polygon_point(t,n)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1_square(x,y)
        else:
            return self.f2_square(x,y)


    def map_to_area(self, t,n, x_min, x_max, y_min, y_max):
        """
        Přemapuje bod z jednotkového čtverce [0,1]x[0,1]
        do obdélníku [x_min, x_max] x [y_min, y_max].
        """
        point = self.hilbert_polygon_point(t,n)
        px, py = point
        new_x = x_min + (x_max - x_min) * px
        new_y = y_min + (y_max - y_min) * py
        return np.array([new_x, new_y])


    def F_mapped(self, t, n, x_min, x_max, y_min, y_max, whatFunc):
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

    def find_minimum_mapped(self, n, x_min, x_max, y_min, y_max, whatFunc, true_min, ftol, maxiter=200):
        if nlopt is None:
            raise ImportError("Package 'nlopt' is required for find_minimum_mapped. Install it with: pip install nlopt")

        if true_min is None:
            raise ValueError("true_min must be provided for ftol stopping in NLOPT.")

        if ftol <= 0:
            raise ValueError("ftol must be positive.")

        def objective(x, grad):
            return self.F_mapped(x[0], n, x_min, x_max, y_min, y_max, whatFunc)

        opt = nlopt.opt(nlopt.GN_DIRECT, 1)
        opt.set_lower_bounds([0.0])
        opt.set_upper_bounds([1.0])
        opt.set_min_objective(objective)
        opt.set_maxeval(maxiter)
        # Stop as soon as the objective reaches the target band around known true minimum.
        # For minimization this corresponds to |f - true_min| < ftol when f >= true_min.
        opt.set_stopval(true_min + ftol)

        t_min = float(opt.optimize([0.5])[0])
        h_min = self.map_to_area(t_min, n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            f_min = self.f(*h_min)
        elif whatFunc == 1:
            f_min = self.f1(*h_min)
        else:
            f_min = self.f2(*h_min)
        nfev = int(opt.get_numevals())
        return t_min, h_min, f_min, nfev




    def differential_evolution_mapped(self, x_min, x_max, y_min, y_max, whatFunc, true_min=None, ftol=1e-6, maxiter=200):
        
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
        N = 2                      
        stop_condition = stop_condition.lower()
        if stop_condition not in {"eps", "ftol"}:
            raise ValueError("stop_condition must be either 'eps' or 'ftol'.")

        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0.0,n, whatFunc), self.F(1.0,n, whatFunc)]
        k = 2
        usedH_arr = []  
        
        # SELECT(2) state variables
        flag = 0  
        imin = 0  
        side_flag = 0  

        for iteracni_krok in range(max_iter):
            
            # STEP 1: serazeni bodu podle hodnoty

            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty
            
            # -----------HOLDER-CONST(1)-----------
            
            if H == -1:
                hvalues = []
                for i in range(1, len(xk)):
                    diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                    hvalues.append(diff)
                h_hat = max(hvalues) 
                h_value = max([h_hat, 1e-8])
                h_used = [h_value] * len(hvalues)
                
                usedH_arr.append(h_value)
                
                
            
            elif H == -2:
                # -----------HOLDER-CONST(2)-----------
                
                
                
                    # Výpočet mi z rovnice (3.3)
                    m_values = []
                    for i in range(1, len(xk)):
                        diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                        m_values.append(diff)
                    
                    # Výpočet λi = max{mi-1, mi, mi+1}
                    lambda_values = []
                    
                    if len(xk) ==2:
                        lambda_values.append(m_values[0])
                    else:      
                        for i in range(1, len(m_values)-1):  
                            lambda_i = max(m_values[i-1], m_values[i], m_values[i+1])
                            lambda_values.append(lambda_i)
                        lambda_2 = max(m_values[0], m_values[1])  # m2, m3
                        lambda_k = max(m_values[-2], m_values[-1])  # mk-1, mk
                        lambda_values = [lambda_2] + lambda_values + [lambda_k]
                    
                    # Výpočet γi = h^k * |xi - xi-1| / X^max
                    gamma_values = []
                    X_max = max([abs(xk[i] - xk[i-1])**(1/N) for i in range(1, len(xk))])
                    
                    
                    h_k = max(m_values)
                    
                    for i in range(1, len(xk)):
                        gamma_i = h_k * abs(xk[i] - xk[i-1])**(1/N) / X_max
                        gamma_values.append(gamma_i)
                    
                    # Výpočet hi = max{λi, γi, ξ}
                    xi_param = 1e-8  # ξ > 0
                    h_values = []
                    for i in range(len(lambda_values)):
                        if i < len(gamma_values):
                            h_i = max(lambda_values[i], gamma_values[i], xi_param)
                           
                        else:
                            h_i = max(lambda_values[i], xi_param)
                        h_values.append(h_i)
                    
                    h_used = h_values
                    temporary = max(h_values) # pro vypocet gammy
                
                
                
                    usedH_arr.append(temporary)
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
            
            # STEP 4: vyber intervalu - implementace SELECT(2)
            if I==1:
                
            #---------- SELECT(1) -----------
                
                idx = np.argmin(Mi)
                y_star = yi[idx]

            else:
                
                #----------- SELECT(2) ------------

                # Initialize idx with standard selection
                idx = np.argmin(Mi)
                
                # Update imin - index odpovídající současnému odhadu minimální hodnoty
                current_imin = np.argmin(zk)
                
                
                if flag == 1:
                    if len(zk) > 1 and zk[-1] < zk[imin]:  
                        imin = len(zk) - 1
                        
                # Local improvement: Alternate the choice of interval
                    delta = 1e-5 
                    
                    # Local improvement logic around imin
                    if imin >= 1 and imin < len(xk)-1:
                       
                        left_size = abs(xk[imin] - xk[imin-1]) if imin > 0 else 0
                        right_size = abs(xk[imin+1] - xk[imin]) if imin+1 < len(xk) else 0
                        
                        # Výběr intervalu podle velikosti a delta - střídání mezi right a left
                        
                        if side_flag == 0:  # nejprve right
                            if right_size > delta and imin < len(Mi):
                                t_choice = imin 
                            elif left_size > delta and imin-1 < len(Mi):
                                t_choice = imin - 1  # interval (x_{imin-1}, x_imin)
                            else:
                                t_choice = np.argmin(Mi)  # standardní výběr
                        else: 
                            if left_size > delta and imin-1 < len(Mi):
                                t_choice = imin - 1  # interval (x_{imin-1}, x_imin)
                            elif right_size > delta and imin < len(Mi):
                                t_choice = imin 
                            else:
                                t_choice = np.argmin(Mi)  # standardní výběr
                        
                        
                        side_flag = 1 - side_flag
                            
                        # Ověření platnosti výběru
                        if 0 <= t_choice < len(Mi):
                            interval_size = abs(xk[t_choice+1] - xk[t_choice])
                            if interval_size > delta:
                                idx = t_choice
                    
                    # Update flag
                    flag = 0  
                else:
                    # Reset flag and use standard selection
                    flag = 1  
                    idx= np.argmin(Mi)
                    imin = current_imin  # Update imin
                
                y_star = yi[idx]
            
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
                    nejjemnejsi_interval = min(
                        abs(xk[i] - xk[i - 1]) ** (1 / N)
                        for i in range(1, len(xk))
                    )
                    if nejjemnejsi_interval < eps:
                        print(f"Algorithm stopped after {iteracni_krok + 1} iterations - smallest interval below threshold.")
                        break
           

            xk.append(y_star)
            zk.append(self.F(y_star, n, whatFunc))
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


   