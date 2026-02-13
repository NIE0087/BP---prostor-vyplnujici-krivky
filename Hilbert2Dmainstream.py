import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, OptimizeResult, minimize
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

# Pomocné funkce pro _minimize_scalar_bounded
def is_finite_scalar(x):
    return np.isfinite(x) and np.isscalar(x)

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        raise TypeError(f"Unknown options: {msg}")

def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        print(f"Optimization terminated successfully.")
    elif flag == 1:
        print(f"Maximum number of function evaluations exceeded: {maxfun}")
    elif flag == 2:
        print(f"NaN encountered.")
    print(f"         Current function value: {fval}")
    print(f"         Iterations: {maxfun}")
    print(f"         Function evaluations: {maxfun}")

_status_message = {'nan': 'NaN result encountered.'}

class Hilbert2Dmainstream:

    def __init__(self, precision: int):
        self.precision = precision


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
    


    def calculate_mainstream_point(self, e0j_counted, dj_counted, q_num,n):
        H0 = np.array([[0, 1],
               [1, 0]])

        H1 = np.array([[1, 0],
               [0, 1]])

        H2 = np.array([[1, 0],
               [0, 1]])

        H3 = np.array([[0, -1],
               [-1, 0]])

        H = [H0, H1, H2, H3]

        F0 = np.array([1/2, 1/2])@ H[0]
        F1 = np.array([1/2, 1/2])@ H[1]
        F2 = np.array([1/2, 1/2])@ H[2]
        F3 = np.array([1/2, 1/2])@ H[3]

        F = [F0, F1, F2, F3]
        soucin = np.eye(2)
        
        q_middle = q_num
        
        
        
        for j in (q_num[:-1]):
            
            soucin = soucin @ H[j]
       
        prvniScitanec = (1/2)**(n) * soucin @ F[q_num[-1]]
      
        s = np.zeros((2, 1))
        for i in range(1, len(q_num) + 1):
            s += (1/(2**i)) * ((-1)**e0j_counted[i-1]) * (
                np.sign(q_num[i-1]) *
                np.array([[(1-dj_counted[i-1])*q_num[i-1]-1],
                          [1-dj_counted[i-1]*q_num[i-1]]])
            )  
       
        s = s.flatten()
       
        d = s + prvniScitanec
        
    

        return  d

    

    def mainstream_hilbert_point(self, t, n):
        
        if t == 1.0:
            q = [3] * n
        else:
            q = self.dec_to_quarter(t)
            q = q[:n]
        
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_mainstream_point(e0, dj, q, n)
        return point.flatten()


    def mainstream_hilbert_polygon_point(self, t, n):
       
        N = 2**(2*n)  
        
    
        if t >= 1.0:
            return self.mainstream_hilbert_point(1.0, n)
        
        k = int(np.floor(t * N))
        
      
        p_k = self.mainstream_hilbert_point(k / N, n)
        p_k1 = self.mainstream_hilbert_point((k + 1) / N, n)
        
     
        point = N * (t - (k/N)) * p_k1 - N * (t - ((k+1)/N)) * p_k
        
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


        return Hilbert2Dmainstream.f1(u,v)
      
    

    # --- Matyas function ---
    @staticmethod
    def f2_square(x, y):
        x_min=-10
        x_max=10
        y_min=-10
        y_max=10
       
        u = x_min + x * (x_max - x_min)
        v = y_min + y * (y_max - y_min)


        return Hilbert2Dmainstream.f2(u,v)
    
    
    #---- Složená funkce -----
    def F(self, t, n, whatFunc):
        x, y = self.mainstream_hilbert_polygon_point(t,n)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1_square(x,y)
        else:
            return self.f2_square(x,y)

    def map_to_area(self, t, n, x_min, x_max, y_min, y_max):
        """
        Přemapuje bod z jednotkového čtverce [0,1]x[0,1]
        do obdélníku [x_min, x_max] x [y_min, y_max]
        pomocí mainstream Hilbertovy křivky.
        """
        point = self.mainstream_hilbert_polygon_point(t, n)
        px, py = point
        new_x = x_min + (x_max - x_min) * px
        new_y = y_min + (y_max - y_min) * py
        return np.array([new_x, new_y])
    
    def F_mapped(self, t, n, x_min, x_max, y_min, y_max, whatFunc):
    
        x, y = self.map_to_area(t, n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1(x, y)
        else:
            return self.f2(x, y)
        

#################################################################
# ------------ ALGORITMY HLEDAJÍCÍ MINIMA ----------------------#
#################################################################



    def _minimize_scalar_bounded(self, func, bounds, true_min, args=(),
                             xatol=1e-5, ftol=1e-6, maxiter=200, disp=0,
                             original_func=None, map_params=None,
                             **unknown_options):
 
        _check_unknown_options(unknown_options)
        maxfun = maxiter
        # Test bounds are of correct form
        if len(bounds) != 2:
            raise ValueError('bounds must have two elements.')
        x1, x2 = bounds

        if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
            raise ValueError("Optimization bounds must be finite scalars.")

        if x1 > x2:
            raise ValueError("The lower bound exceeds the upper bound.")

        flag = 0
        header = ' Func-count     x          f(x)          Procedure'
        step = '       initial'

        sqrt_eps = sqrt(2.2e-16)
        golden_mean = 0.5 * (3.0 - sqrt(5.0))
        a, b = x1, x2
        fulc = a + golden_mean * (b - a)
        nfc, xf = fulc, fulc
        rat = e = 0.0
        x = xf
        fx = func(x, *args)
        num = 1
        fmin_data = (1, xf, fx)
        fu = np.inf

        ffulc = fnfc = fx
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if disp > 2:
            print(" ")
            print(header)
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        # Výpočet hodnoty původní funkce pro porovnání
        if true_min is not None and original_func is not None and map_params is not None:
            n, x_min, x_max, y_min, y_max, whatFunc = map_params
            point = self.map_to_area(xf, n, x_min, x_max, y_min, y_max)
            if whatFunc == 0:
                original_fx = self.f(*point)
            elif whatFunc == 1:
                original_fx = self.f1(*point)
            else:
                original_fx = self.f2(*point)
        else:
            original_fx = fx

        while (true_min is None or np.abs(original_fx - true_min) >= ftol):
            golden = 1
            # Check for parabolic fit
            if np.abs(e) > tol1:
                golden = 0
                r = (xf - nfc) * (fx - ffulc)
                q = (xf - fulc) * (fx - fnfc)
                p = (xf - fulc) * q - (xf - nfc) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = np.abs(q)
                r = e
                e = rat

                # Check for acceptability of parabola
                if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                        (p < q * (b - xf))):
                    rat = (p + 0.0) / q
                    x = xf + rat
                    step = '       parabolic'

                    if ((x - a) < tol2) or ((b - x) < tol2):
                        si = np.sign(xm - xf) + ((xm - xf) == 0)
                        rat = tol1 * si
                else:      # do a golden-section step
                    golden = 1

            if golden:  # do a golden-section step
                if xf >= xm:
                    e = a - xf
                else:
                    e = b - xf
                rat = golden_mean*e
                step = '       golden'

            si = np.sign(rat) + (rat == 0)
            x = xf + si * np.maximum(np.abs(rat), tol1)
            fu = func(x, *args)
            num += 1
            fmin_data = (num, x, fu)
            if disp > 2:
                print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

            if fu <= fx:
                if x >= xf:
                    a = xf
                else:
                    b = xf
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = xf, fx
                xf, fx = x, fu
            else:
                if x < xf:
                    a = x
                else:
                    b = x
                if (fu <= fnfc) or (nfc == xf):
                    fulc, ffulc = nfc, fnfc
                    nfc, fnfc = x, fu
                elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                    fulc, ffulc = x, fu

            xm = 0.5 * (a + b)
            tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
            tol2 = 2.0 * tol1

            # Přepočet hodnoty původní funkce pro aktuální xf
            if true_min is not None and original_func is not None and map_params is not None:
                n, x_min, x_max, y_min, y_max, whatFunc = map_params
                point = self.map_to_area(xf, n, x_min, x_max, y_min, y_max)
                if whatFunc == 0:
                    original_fx = self.f(*point)
                elif whatFunc == 1:
                    original_fx = self.f1(*point)
                else:
                    original_fx = self.f2(*point)

            if num >= maxfun:
                flag = 1
                break

        if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
            flag = 2

        fval = fx
        if disp > 0:
            _endprint(x, flag, fval, maxfun, xatol, disp)

        result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                                message={0: 'Solution found.',
                                         1: 'Maximum number of function calls '
                                            'reached.',
                                         2: _status_message['nan']}.get(flag, ''),
                                x=xf, nfev=num, nit=num)

        return result





    def find_minimum_mapped(self,n, x_min, x_max, y_min, y_max, whatFunc,true_min, ftol, maxiter=200):
        map_params = (n, x_min, x_max, y_min, y_max, whatFunc)
        result = self._minimize_scalar_bounded(
            lambda t: self.F_mapped(t,n, x_min, x_max, y_min, y_max, whatFunc), 
            bounds=(0, 1), 
            true_min=true_min, 
            ftol=ftol,
            maxiter=maxiter,
            original_func=True,
            map_params=map_params
        )
        t_min = result.x
        h_min = self.map_to_area(t_min,n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            f_min = self.f(*h_min)
        elif whatFunc == 1:
            f_min = self.f1(*h_min)
        else:
            f_min = self.f2(*h_min)
        nfev = result.nfev  # Počet vyhodnocení funkce
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





    def Holder_algorithm_mapped(self,H,I, r,eps,max_iter,n, whatFunc, true_min, ftol):
        N = 2                      
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
                        gamma_i = h_k * abs(xk[i] - xk[i-1]) / X_max
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
            x_current, y_current = self.mainstream_hilbert_polygon_point(t_current,n)
            
            if whatFunc == 0:
                f_current = self.f(x_current, y_current)
            elif whatFunc == 1:
                f_current = self.f1_square(x_current, y_current)
            else:
                f_current = self.f2_square(x_current, y_current)
            
            # Kontrola přesnosti od skutečného minima
            if true_min is not None and abs(f_current - true_min) < ftol:
                print(f"Algorithm stopped after {iteracni_krok} iterations - desired accuracy achieved.")
                break

           
            xk.append(y_star)
            zk.append(self.F(y_star, n, whatFunc))
            k += 1
        
        min_idx = np.argmin(zk)
        t_min = xk[min_idx]               # parametr t na Hilbertově křivce
        x_min_mapped, y_min_mapped = self.mainstream_hilbert_polygon_point(t_min,n)  # souřadnice v R^2
        if whatFunc==0:
            f_min = self.f(x_min_mapped, y_min_mapped)      # hodnota funkce f(x,y)
        elif whatFunc==1:
            f_min = self.f1_square(x_min_mapped, y_min_mapped)
        else:
            f_min = self.f2_square(x_min_mapped,y_min_mapped)
    
        return t_min, f_min, x_min_mapped, y_min_mapped, usedH_arr



