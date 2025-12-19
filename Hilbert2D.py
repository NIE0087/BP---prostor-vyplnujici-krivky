import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

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


    
    # --------------- Mainstreamova verze ----------------



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
        
        
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_mainstream_point(e0, dj, q, n)
        return point.flatten()

    

    
    

    

#################################################################
# ------------ ZKUŠEBNÍ FUNKCE PRO OPTIMALIZACI-----------------#
#################################################################



    @staticmethod
    # --- Random function ---
    def f(x, y):
        return ((x - 0.3)**2 + (y - 0.7)**2)**1/2 + 1
    
    
    # --- Himmelblau's function --- 
    # --- Three-hump camel function ---
    # --- Easom function ---
    @staticmethod
    def f1(x, y):
        return (x**2 + y - 11)**2 + (y**2 + x - 7)**2
        #return 2*x**2 - 1.05*x**4 + (x**6)/6 + y*x + y**2
        #return -math.cos(x)*math.cos(y)*math.exp(-((x-math.pi)**2 + (y-math.pi)**2))
    

    # --- Matyas function ---
    @staticmethod
    def f2(x, y):
        return 0.26*(x**2 + y**2) - 0.48*y*x
    



    #---- Složená funkce -----
    def F(self, t, n, whatFunc):
        x, y = self.hilbert_polygon_point(t,n)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1(x,y)
        else:
            return self.f2(x,y)
     



#################################################################
# --------------------- MAPOVÁNÍ FUNKCÍ-------------------------#
#################################################################



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
        x, y = self.map_to_area(t,n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            return self.f(x, y)
        elif whatFunc == 1:
            return self.f1(x,y)
        else:
            return self.f2(x,y)
    

#################################################################
# ------------ ALGORITMY HLEDAJÍCÍ MINIMA ----------------------#
#################################################################




    def find_minimum_mapped(self,n, x_min, x_max, y_min, y_max, whatFunc):
        result = minimize_scalar(lambda t: self.F_mapped(t,n, x_min, x_max, y_min, y_max, whatFunc), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.map_to_area(t_min,n, x_min, x_max, y_min, y_max)
        if whatFunc == 0:
            f_min = self.f(*h_min)
        elif whatFunc == 1:
            f_min = self.f1(*h_min)
        else:
            f_min = self.f2(*h_min)
        return t_min, h_min, f_min




    def differential_evolution_mapped(self, x_min, x_max, y_min, y_max, whatFunc, maxiter=30):
        
        def objective(coords):
            x, y = coords
            if whatFunc == 0:
                return self.f(x, y)
            elif whatFunc == 1:
                return self.f1(x, y)
            else:
                return self.f2(x, y)
        
        bounds = [(x_min, x_max), (y_min, y_max)]
        
        result = differential_evolution(objective, bounds, maxiter=maxiter)
        
        x_min_de, y_min_de = result.x
        f_min = result.fun
        
        return f_min, x_min_de, y_min_de





    def Holder_algorithm_mapped(self,H,I, r,eps,max_iter,n,x_min, x_max, y_min, y_max, whatFunc):
        N = 2                      
        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F_mapped(0.0,n, x_min, x_max, y_min, y_max, whatFunc), self.F_mapped(1.0,n, x_min, x_max, y_min, y_max, whatFunc)]
        k = 2
        usedH_arr = []  
        
        # SELECT(2) state variables
        flag = 1  
        imin = 0  

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
                h_hat = max(hvalues) if hvalues else H
                h_used = max([h_hat, 1e-8])   
                usedH_arr.append(h_used)
                
            
            elif H == -2:
                # -----------HOLDER-CONST(2)-----------
                
                if len(xk) < 3:
                    # Pro méně než 3 body 
                    hvalues = []
                    for i in range(1, len(xk)):
                        diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                        hvalues.append(diff)
                    h_hat = max(hvalues) if hvalues else 1e-8
                    h_used = max([h_hat, 1e-8])
                
                else:
                    # Výpočet mi z rovnice (3.3)
                    m_values = []
                    for i in range(1, len(xk)):
                        diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                        m_values.append(diff)
                    
                    # Výpočet λi = max{mi-1, mi, mi+1}
                    lambda_values = []
                    for i in range(1, len(m_values)-1):  
                        lambda_i = max(m_values[i-1], m_values[i], m_values[i+1])
                        lambda_values.append(lambda_i)
                    
                    # Pro i=2 a i=k bereme jen odpovídající m hodnoty
                    if len(m_values) == 2 or len(m_values) == max_iter:
                        lambda_2 = max(m_values[0], m_values[1])  # m2, m3
                        lambda_k = max(m_values[-2], m_values[-1])  # mk-1, mk
                        lambda_values = [lambda_2] + lambda_values + [lambda_k]
                    
                    # Výpočet γi = h^k * |xi - xi-1| / X^max
                    gamma_values = []
                    X_max = max([abs(xk[i] - xk[i-1])**(1/N) for i in range(1, len(xk))])
                    
                    if iteracni_krok > 0:  # h^k z předchozí iterace
                        h_k = usedH_arr[-1] if usedH_arr else 1e-8
                    
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
                    
                    h_used = max(h_values) if h_values else 1e-8
                   
                
                
                
                usedH_arr.append(h_used)
            else:
                h_used = H
                
                usedH_arr.append(h_used)
          
            # STEP 3: vypocet pruseciku a M_i
            Mi = []
            yi = []
            for i in range(1, len(xk)):
            
                y = 0.5*(xk[i-1] + xk[i]) - (zk[i] - zk[i-1])/(2*r*h_used*(xk[i]-xk[i-1])**((1-N)/N))
                yi.append(y)
                # Vypocet M_i 
                Mi.append(min(zk[i-1] - r*h_used * abs(y - xk[i-1])**(1/N), zk[i] - r*h_used * abs(xk[i] - y)**(1/N)))
            
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
                if flag == 0:
                    delta = 1e-5 # δ > 0 threshold
                    
                    # Local improvement logic around imin
                    if imin >= 1 and imin < len(xk)-1:
                        # Kontrola velikosti intervalů kolem imin
                        left_size = abs(xk[imin] - xk[imin-1]) if imin > 0 else 0
                        right_size = abs(xk[imin+1] - xk[imin]) if imin+1 < len(xk) else 0
                        
                        # Výběr intervalu podle velikosti a delta
                        
                        if right_size > delta and imin < len(Mi):
                         
                            t_choice = imin 
                        
                        elif left_size > delta and imin-1 < len(Mi):
                   
                            t_choice = imin - 1  # interval (x_{imin-1}, x_imin)
                     

                        else:
                           
                            t_choice = np.argmin(Mi)  # standardní výběr
                            
                        # Ověření platnosti výběru
                        if 0 <= t_choice < len(Mi):
                            interval_size = abs(xk[t_choice+1] - xk[t_choice])
                            if interval_size > delta:
                                idx = t_choice
                    
                    # Update flag
                    flag = 1  
                else:
                    # Reset flag and use standard selection
                    flag = 0  
                    imin = current_imin  # Update imin
                
                y_star = yi[idx]
            
            # STEP 5: zastavovaci podminka - kontrolujeme velikost vybraného intervalu
            interval_length = abs(xk[idx+1] - xk[idx]) if idx+1 < len(xk) else float('inf')
            if interval_length**(1/N) < eps:
                print(f"Algorithm stopped after {iteracni_krok} iterations.")
                break

           
            xk.append(y_star)
            zk.append(self.F_mapped(y_star, n, x_min, x_max, y_min, y_max, whatFunc))
            k += 1
        
        min_idx = np.argmin(zk)
        t_min = xk[min_idx]               # parametr t na Hilbertově křivce
        x_min_mapped, y_min_mapped = self.map_to_area(t_min, n,x_min, x_max, y_min, y_max)  # souřadnice v R^2
        if whatFunc==0:
            f_min = self.f(x_min_mapped, y_min_mapped)      # hodnota funkce f(x,y)
        elif whatFunc==1:
            f_min = self.f1(x_min_mapped, y_min_mapped)
        else:
            f_min = self.f2(x_min_mapped,y_min_mapped)
        return t_min, f_min, x_min_mapped, y_min_mapped, usedH_arr






#################################################################
# ------------ NEDŮLEŽITÉ VĚCI - POUZE GRAFY A TABULKY ---------#
#################################################################






















    def plot_hilbert_polygon(self, n):
        

        N = 2**(2*n)
        points = []

        for k in range(N+1):
            t = k / N
            points.append(self.hilbert_polygon_point(t,n))

        points = np.array(points)
        plt.plot(points[:,0], points[:,1], '-o', markersize=2)
        plt.axis('equal')
        plt.show()

    # --- graf pro mainstreamovou podobu ---
    def plot_mainstream_hilbert_polygon(self, n):
        

        points = []
        for k in range(4 ** n):
            t = k / (4 ** n)
            p = self.mainstream_hilbert_point(t,n)
            points.append(p)

        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], '-o', markersize=2)
        plt.axis('equal')
        plt.show()

    def plot_multiple_hilberts(self, orders):
        
            fig, axes = plt.subplots(1, len(orders), figsize=(4*len(orders), 4))

            if len(orders) == 1:
                axes = [axes]

            for ax, n in zip(axes, orders):
       
                points = []
                for k in range(4 ** n):
                 t = k / (4 ** n)
                 p = self.mainstream_hilbert_point(t, n)
                 points.append(p)
                points = np.array(points)

       
                ax.plot(points[:, 0], points[:, 1], '-o', markersize=2)
                ax.set_aspect("equal")
                ax.set_title(f"n = {n}")
                ax.axis("on")

       
                step = 1 / (2**n)   
                for i in range(2**n+1):
                    for j in range(2**n+1):
                       square = patches.Rectangle(
                            (i*step, j*step), step, step,
                            facecolor="white",   
                            edgecolor="black",      
                            linewidth=0.5
                        )
                       ax.add_patch(square)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect("equal")
                    ax.set_title(f"n = {n}")
                    ax.axis("on")
  
            
            plt.tight_layout()
            plt.show()

    def plot_multiple_hilberts_arrows(self, orders):

     fig, axes = plt.subplots(1, len(orders), figsize=(4*len(orders), 4))

     if len(orders) == 1:
        axes = [axes]

     for ax, n in zip(axes, orders):
        
        if n == 0:
           
            points = np.array([[0,0], [1,0]])

        elif n == 1:
           
            points = np.array([
                [0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [1.0, 0.0]   
                
            ])

        else:
            
            points = []
            for k in range(4 ** n+1):
                t = k / (4 ** n)
                p = self.hilbert_polygon_point(t, n)
                points.append(p)
            points = np.array(points)


        ax.plot(points[:, 0], points[:, 1], '-o', markersize=2)

        #  šipky
        for i in range(len(points)-1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            ax.annotate("",
                        xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

        # mřížka
        step = 1 / (2**max(1,n))   
        for i in range(2**max(1,n)+1):
            for j in range(2**max(1,n)+1):
                square = patches.Rectangle(
                    (i*step, j*step), step, step,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.5
                )
                ax.add_patch(square)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"({n})")
        ax.axis("on")

     plt.tight_layout()
     plt.show()







    def compare_algorithms(self, H, I, r, eps, max_iter, N_vals, x_min, x_max, y_min, y_max, whatFunc, true_min):
     
        results = []
        
        for n in N_vals:
          
            _, f_m, _, _, _ = self.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, x_min, x_max, y_min, y_max, whatFunc)
            diff_m = abs(f_m - true_min)

       
            _,_,f_nm = self.find_minimum_mapped(n, x_min, x_max, y_min, y_max, whatFunc)
            diff_nm = abs(f_nm - true_min)
            
            results.append([n, f_m, diff_m, f_nm, diff_nm])

        df = pd.DataFrame(results, columns=["Iterace n", "Hodnota Hoelder", "Rozdíl Hoelder", "Hodnota scipy", "Rozdíl scipy"])
        print(df[["Iterace n", "Rozdíl Hoelder", "Rozdíl scipy"]])
        n_arr = df["Iterace n"].to_numpy()
        holder_arr = df["Rozdíl Hoelder"].to_numpy()
        scipy_arr = df["Rozdíl scipy"].to_numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_arr, holder_arr, 'o-', label="Hölder algoritmus")
        plt.plot(n_arr, scipy_arr, 's-', label="Scipy (Hilbert)")
        plt.xlabel("Iterace Hilbertovy křivky (n)")
        plt.ylabel("Rozdíl od opravdového minima")
        plt.title("Porovnání optimalizačních algoritmů")
        plt.yscale("log")  
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.show()
        
        return df

    def compare_holder_variants(self, r, eps, max_iter, N_vals, x_min, x_max, y_min, y_max, whatFunc, true_min):
        """
        Porovná různé varianty Hölderova algoritmu:
        H=-1 vs H=-2 a I=1 vs I=2
        """
        variants = [
            ("H=-1, I=1", -1, 1),
            ("H=-1, I=2", -1, 2), 
            ("H=-2, I=1", -2, 1),
            ("H=-2, I=2", -2, 2)
        ]
        
        # Slovníky pro uložení výsledků
        results = {name: [] for name, _, _ in variants}
        n_values = []
        
        for n in N_vals:
            n_values.append(n)
            for name, H, I in variants:
                _, f_min, _, _, _ = self.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, x_min, x_max, y_min, y_max, whatFunc)
                diff = abs(f_min - true_min)
                results[name].append(diff)
        
        # Vykreslí graf
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green', 'orange']
        markers = ['o', 's', '^', 'v']
        
        for i, (name, _, _) in enumerate(variants):
            plt.plot(n_values, results[name], color=colors[i], marker=markers[i], 
                    label=name, linewidth=2, markersize=4)
        
        plt.xlabel("Iterace Hilbertovy křivky (n)")
        plt.ylabel("Rozdíl od opravdového minima")
        plt.title("Porovnání variant Hölderova algoritmu", fontsize=14)
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10, frameon=True, fancybox=True)
        plt.tight_layout()
        plt.show()
        

    def hyperparameter_tuning_r(self, r_values, H, I, eps, max_iter, N_vals, x_min, x_max, y_min, y_max, whatFunc, true_min):
        """
        Hyperparameter tuning pro parametr r
        """
        results = {f"r={r}": [] for r in r_values}
        n_values = []
        
        for n in N_vals:
            n_values.append(n)
            for r in r_values:
                _, f_min, _, _, _ = self.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, x_min, x_max, y_min, y_max, whatFunc)
                diff = abs(f_min - true_min)
                results[f"r={r}"].append(diff)
        
        # Vykreslí graf
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
        
        for i, r in enumerate(r_values):
            marker = markers[i % len(markers)]
            plt.plot(n_values, results[f"r={r}"], 
                    color=colors[i], 
                    marker=marker, 
                    linestyle='-',
                    label=f"r={r}", 
                    linewidth=2, 
                    markersize=6,
                    markerfacecolor='white',
                    markeredgecolor=colors[i],
                    markeredgewidth=2)
        
        plt.xlabel("Iterace Hilbertovy křivky (n)", fontsize=12)
        plt.ylabel("Rozdíl od opravdového minima", fontsize=12)
        plt.title(f"Hyperparameter tuning r (H={H}, I={I})", fontsize=14)
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10, frameon=True, fancybox=True, ncol=2)
        plt.tight_layout()
        plt.show()
        






    def analyze_holder_constants(self, H_true,I,H, r, eps, max_iter, N_vals, x_min, x_max, y_min, y_max, whatFunc):
    
        results = []
        
        for n in N_vals:
            _, _, _, _, usedH_arr = self.Holder_algorithm_mapped(H,I, r, eps, max_iter, n, x_min, x_max, y_min, y_max, whatFunc)
            
            if usedH_arr:
                h_final = usedH_arr[-1]
                h_mean = np.mean(usedH_arr)
                results.append([n, H_true, h_mean, h_final])
            else:
                results.append([n, H_true, 0, 0])
        
        df = pd.DataFrame(results, columns=["n", "H opravdové", "H průměr", "H finální"])
        print(df)
        
        return df



    # ------------ Vykreslování paraboloidů ---------



    def plot_holder_paraboloids(self, H, r, eps, max_iter, n, x_min, x_max, y_min, y_max, whatFunc, iteration_to_plot=0):
    
        N = 2
        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F_mapped(0.0, n, x_min, x_max, y_min, y_max, whatFunc), 
              self.F_mapped(1.0, n, x_min, x_max, y_min, y_max, whatFunc)]
        
        current_iteration = 0
        usedH_viz = []  # Pro sledování H hodnot ve vizualizaci
        
        for iteracni_krok in range(max_iter):
            # STEP 1: serazeni bodu podle hodnoty
            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty pro paraboloidy (stejná logika jako v hlavním algoritmu)
            if H == -1:
                hvalues = []
                for i in range(1, len(xk)):
                    diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                    hvalues.append(diff)
                h_used = max(hvalues) if hvalues else 1e-8
                h_used = max([h_used, 1e-8])
                usedH_viz.append(h_used)
            elif H == -2:
                # HOLDER-CONST(2) implementace
                if len(xk) < 3:
                    hvalues = []
                    for i in range(1, len(xk)):
                        diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                        hvalues.append(diff)
                    h_used = max(hvalues) if hvalues else 1e-8
                    h_used = max([h_used, 1e-8])
                else:
                    # Stejná implementace jako v hlavním algoritmu
                    m_values = []
                    for i in range(1, len(xk)):
                        diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                        m_values.append(diff)
                    
                    lambda_values = []
                    for i in range(1, len(m_values)-1):
                        lambda_i = max(m_values[i-1], m_values[i], m_values[i+1])
                        lambda_values.append(lambda_i)
                    
                    if len(m_values) >= 2:
                        lambda_2 = max(m_values[0], m_values[1])
                        lambda_k = max(m_values[-2], m_values[-1])
                        lambda_values = [lambda_2] + lambda_values + [lambda_k]
                    
                    gamma_values = []
                    X_max = max([abs(xk[i] - xk[i-1])**(1/N) for i in range(1, len(xk))])
                    
                    if iteracni_krok > 0:
                        h_k = usedH_viz[-1] if usedH_viz else 1e-8
                    else:
                        h_k = 1e-8
                    
                    for i in range(1, len(xk)):
                        gamma_i = h_k * abs(xk[i] - xk[i-1]) / X_max
                        gamma_values.append(gamma_i)
                    
                    xi_param = 1e-8
                    h_values = []
                    for i in range(len(lambda_values)):
                        if i < len(gamma_values):
                            h_i = max(lambda_values[i], gamma_values[i], xi_param)
                        else:
                            h_i = max(lambda_values[i], xi_param)
                        h_values.append(h_i)
                    
                    h_used = max(h_values) if h_values else 1e-8
                
                usedH_viz.append(h_used)
            else:
                hvalues = []
                for i in range(1, len(xk)):
                   diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
                   hvalues.append(diff)
                h_used = max(hvalues) if hvalues else H
                h_used = max([h_used, 1e-8])
                usedH_viz.append(h_used)


            
            # Pokud jsme na požadované iteraci, vykreslíme
            if current_iteration == iteration_to_plot:
                self._plot_paraboloids_at_iteration(xk, zk, h_used, r, N, n, x_min, x_max, y_min, y_max, whatFunc, current_iteration)
                return
            
            # STEP 3: vypocet pruseciku a M_i
            Mi = []
            yi = []
            for i in range(1, len(xk)):
                y = 0.5*(xk[i-1] + xk[i]) - (zk[i] - zk[i-1])/(2*r*h_used*(xk[i]-xk[i-1])**((1-N)/N))

                yi.append(y)
                Mi.append(min(zk[i-1] - r*h_used * abs(y - xk[i-1])**(1/N), 
                             zk[i] - r*h_used * abs(xk[i] - y)**(1/N)))
            
            # STEP 4: vyber intervalu
            idx = np.argmin(Mi)
            y_star = yi[idx]
            
            # STEP 5: zastavovaci podminka
            if abs(xk[idx+1] - xk[idx])**(1/N) < eps:
                break
            
            xk.append(y_star)
            zk.append(self.F_mapped(y_star, n, x_min, x_max, y_min, y_max, whatFunc))
            current_iteration += 1

    
    
    
    
    
    def _plot_paraboloids_at_iteration(self, xk, zk, h_used, r, N, n, x_min, x_max, y_min, y_max, whatFunc, iteration):
     
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Levý graf: 1D průběh funkce na Hilbertově křivce + paraboloidy

        P = 2**(2*n)
        curve_points = []
        t_values = []
        for k in range(P+1):
            t = k / P
            t_values.append(t)
            curve_points.append(self.map_to_area(t, n, x_min, x_max, y_min, y_max))
        curve_points = np.array(curve_points)
        f_dense = [self.F_mapped(t, n, x_min, x_max, y_min, y_max, whatFunc) for t in t_values]
        
        ax1.plot(t_values, f_dense, 'b-', alpha=0.7, label='F(t) na Hilbertově křivce')
        ax1.scatter(xk, zk, color='red', s=50, zorder=5, label='Známé body')
        
    
        colors = ['orange', 'green', 'purple', 'brown', 'pink']
        for i in range(1, len(xk)):
            x1, z1 = xk[i-1], zk[i-1]
            x2, z2 = xk[i], zk[i]
            
        
            y_intersect = 0.5*(x1 + x2) - (z2 - z1)/(2*r*h_used*(x2-x1)**((1-N)/N))
        
            
        
            t_interval = np.linspace(x1, x2, 190)
            parab1 = z1 - r*h_used * np.abs(t_interval - x1)**(1/N)  # z levého bodu
            parab2 = z2 - r*h_used * np.abs(t_interval - x2)**(1/N)  # z pravého bodu
        
            
            interval_factor = (x2 - x1)**((1-N)/N)
      
            line1 = (-r*h_used * interval_factor * t_interval + 
                    r*h_used * interval_factor * x1 + z1)
            
      
            line2 = (r*h_used * interval_factor * t_interval - 
                    r*h_used * interval_factor * x2 + z2)


            color = colors[i % len(colors)]
            
            ax1.plot(t_interval, parab1, '--', color=color, alpha=0.9, linewidth=2,
                    label=f'$ri_n$ (z bodu {i-1})' if i <= 3 else '')
            ax1.plot(t_interval, parab2, ':', color=color, alpha=0.9, linewidth=2)
            
          
            ax1.plot(t_interval, line1, '-', color='darkgray', alpha=0.7, linewidth=1,
                    label='$li_n$' if i == 1 else '')
            ax1.plot(t_interval, line2, '-', color='darkgray', alpha=0.7, linewidth=1)
            
         
            if x1 <= y_intersect <= x2:
                z_intersect = self.F_mapped(y_intersect, n, x_min, x_max, y_min, y_max, whatFunc)
                
            
                Mi_value = min(z1 - r*h_used * abs(y_intersect - x1)**(1/N), 
                              z2 - r*h_used * abs(x2 - y_intersect)**(1/N))
                
                ax1.scatter([y_intersect], [z_intersect], color=color, s=100, marker='x', zorder=10, 
                           label='F(yi)' if i == 1 else '')
             
                ax1.scatter([y_intersect], [Mi_value], color='pink', s=80, marker='o', zorder=15,
                           label='Mi' if i == 1 else '')
                
                ax1.axvline(x=y_intersect, color=color, linestyle=':', alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('t')
        ax1.set_ylabel('F(t)')
        ax1.set_title(f'Iterace {iteration}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pravý graf: 2D zobrazení bodů v původním prostoru
        points_2d = [self.map_to_area(t, n, x_min, x_max, y_min, y_max) for t in xk]
        points_2d = np.array(points_2d)
        
        ax2.scatter(points_2d[:, 0], points_2d[:, 1], color='red', s=50, zorder=5)
        for i, (x, y) in enumerate(points_2d):
            ax2.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Hilbertova křivka
       
        curve_points = []

        for k in range(P+1):
            t = k / P
            curve_points.append(self.map_to_area(t, n, x_min, x_max, y_min, y_max))

        curve_points = np.array(curve_points) 

        
        ax2.plot(curve_points[:, 0], curve_points[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Iterace {iteration}: Body v 2D prostoru')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()






