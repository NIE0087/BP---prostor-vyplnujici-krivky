import math
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Hilbert2D:

    def __init__(self, precision: int):
        self.precision = precision

    # --- Konverze ---
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
    
    def calculate_nicer_point(self, e0j_counted, dj_counted, q_num):
        H0 = np.array([[0, 1],
               [1, 0]])

        H1 = np.array([[1, 0],
               [0, 1]])

        H2 = np.array([[1, 0],
               [0, 1]])

        H3 = np.array([[0, -1],
               [-1, 0]])

        H = [H0, H1, H2, H3]

        F0 = np.array([1/4, 1/4])
        F1 = np.array([1/4, 3/4])
        F2 = np.array([3/4, 3/4])
        F3 = np.array([3/4, 1/4])

        F = [F0, F1, F2, F3]
        

        soucin = np.eye(2)
        for j in (q_num[:-1]):
            
            soucin = soucin @ H[j]

        s = np.zeros((2, 1))
        for i in range(1, len(q_num)):
            s += (1/(2**i)) * ((-1)**e0j_counted[i-1]) * (
                np.sign(q_num[i-1]) *
                np.array([[(1-dj_counted[i-1])*q_num[i-1]-1],
                          [1-dj_counted[i-1]*q_num[i-1]]])
            )    
       
        s=s.flatten()
        d=(((1/2) * soucin) @ F[q_num[-1]]) + s
        print(d)

        return  (1/2) * soucin @ F[q_num[-1]] + s

        

    def hilbert_point(self, t):
        if t == 1.0:
            return np.array([1.0, 0.0]) 
        
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_point(e0, dj, q)
        return point.flatten()
    

    def nicer_hilbert_point(self, t):
        if t == 1.0:
            return np.array([1.0, 0.0]) 
        
        q = self.dec_to_quarter(t)
        e0, dj = self.ej_and_dj_counter(q)
        point = self.calculate_nicer_point(e0, dj, q)
        return point.flatten()

    
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


    def plot_nicer_hilbert_polygon(self, n):
        

        points = []
        for k in range(4 ** n):
            t = k / (4 ** n)
            p = self.nicer_hilbert_point(t)
            points.append(p)

        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], '-o', markersize=2)
        plt.axis('equal')
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


class Hilbert3D:
    
    def __init__(self, precision: int):
        self.precision = precision

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
        elif number == 0:
            q_num.append(0)
        
        elif number == 1.0:
            q_num.append(1)
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
    


    def plot_hilbert_curve(self,n):
        samples = 8**n
        pts = np.zeros((samples, 3))

        for k in range(samples):
            t= k/samples
            q = self.dec_to_octal(t)  
            pts[k] = self.ThreeD_Hilbert(q)  

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color='purple', linewidth=0.5) 
        ax.set_box_aspect([1,1,1])
        plt.show()