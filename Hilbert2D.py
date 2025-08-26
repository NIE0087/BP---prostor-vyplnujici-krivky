import math
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class Hilbert2D:

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
    
    # --- Mainstreamova verze ---

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

    # --- PLOTS ---
    
    def hilbert_polygon_point(self, t, n):
    
        N = 2**(2*n)
    
    
        k = int(np.floor(t * N))
        #if t == 1.0: chaby pokus 
        #   t = 1.0 - 1e-12

        p_k = self.hilbert_point(k / N)
        p_k1 = self.hilbert_point((k + 1) / N)
        
        point = N * (t-(k/N))*p_k1 - N*(t-((k+1)/N))*p_k

        return point
    
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
            # ---- n=0, jen úsečka
            points = np.array([[0,0], [1,0]])

        elif n == 1:
            # ---- n=1, ručně zadaná Hilbertova křivka
            points = np.array([
                [0.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [1.0, 0.0],
                
                
            ])

        else:
            # ---- ostatní řády podle generátoru
            points = []
            for k in range(4 ** n+1):
                t = k / (4 ** n)
                p = self.hilbert_polygon_point(t, n)
                points.append(p)
            points = np.array(points)





        # vykreslit body + čáru
        ax.plot(points[:, 0], points[:, 1], '-o', markersize=2)

        # přidat šipky
        for i in range(len(points)-1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            ax.annotate("",
                        xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

        # mřížka
        step = 1 / (2**max(1,n))   # aby to fungovalo i pro n=0
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


    # --- Optimalizace ---
    @staticmethod
    def f(x, y):
        return ((x - 0.3)**2 + (y - 0.7)**2)**1/2 + 1

    def F(self, t, n):
        x, y = self.hilbert_polygon_point(t,n)
        return self.f(x, y)
   
    # --- Zabudovana python metoda pro hledani minima ---
   
    def find_minimum(self,n):
        result = minimize_scalar(lambda t: self.F(t,n), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.hilbert_point(t_min)
        f_min = self.f(*h_min)
        return t_min, h_min, f_min
   
    # --- Optimalizacni algoritmus pro hledani minima ---
    def Holder_algorithm(self,H,r,eps,max_iter,n):
        N = 2                      
        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0.0,n), self.F(1.0,n)]
        k = 2

        for iteracni_krok in range(max_iter):
            
            # STEP 1: serazeni bodu podle hodnoty

            xk, zk = (list(t) for t in zip(*sorted(zip(xk, zk))))

            # STEP 2: odhad Holderovy konstanty
            hvalues = []
            for i in range(1, len(xk)):
               diff = abs(zk[i] - zk[i-1]) / (abs(xk[i] - xk[i-1]))**(1/N) if abs(xk[i]-xk[i-1])>0 else 0
               hvalues.append(diff)
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
            zk.append(self.F(y_star,n))
            k += 1
        
        min_idx = np.argmin(zk)
        t_min = xk[min_idx]               # parametr t na Hilbertově křivce
        x_min, y_min = self.hilbert_point(t_min)  # souřadnice v R^2
        f_min = self.f(x_min, y_min)      # hodnota funkce f(x,y)

        return t_min, f_min, x_min, y_min

