import math
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    
    
   
    def plot_hilbert_polygon(self,n):
        N = 8**n
        pts = np.zeros((N, 3))

        for k in range(N):
            t= k/N
            pts[k] = self.hilbert_polygon_point(t,n)  

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color='purple', linewidth=0.5) 
        ax.set_box_aspect([1,1,1])
        plt.show()


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
    
    def plot_mainstream_hilbert(self,n):
        samples = 8**n
        pts = np.zeros((samples, 3))

        for k in range(samples):
            t= k/samples
            q = self.dec_to_octal(t)  
            pts[k] = self.calculate_mainstream_point(q,n)  

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color='purple', linewidth=0.5) 
        ax.set_box_aspect([1,1,1])
        plt.show()

    def plot_mainstream_hilbert_cubes(self, n):
     samples = 8**n
     ts = np.zeros((samples, 3))
     pts = np.zeros((samples, 3))
     for k in range(samples):
            t = k/samples
            q = self.dec_to_octal(t)  
            pts[k] = self.calculate_mainstream_point(q, n)  

     fig = plt.figure(figsize=(6,6))
     ax = fig.add_subplot(projection='3d')
     ax.grid(False)
     ax.plot(pts[:,0], pts[:,1], pts[:,2], color='purple', linewidth=0.5) 
     ax.set_box_aspect([1,1,1])


    # === Přidání mřížky ===
     divs = 2**n   # počet dělení os
     grid = np.linspace(0, 1, divs+1)  # rozdělení 0..1

    # Čáry rovnoběžné s osou x
     for y in grid:
        for z in grid:
            ax.plot([0,1], [y,y], [z,z], color="black", linewidth=0.9, alpha=0.5)

    # Čáry rovnoběžné s osou y
     for x in grid:
        for z in grid:
            ax.plot([x,x], [0,1], [z,z], color="black", linewidth=0.9, alpha=0.5)

    # Čáry rovnoběžné s osou z
     for x in grid:
        for y in grid:
         ax.plot([x,x], [y,y], [0,1], color="black", linewidth=0.9, alpha=0.5)
   
         ax.scatter(pts[:,0], pts[:,1], pts[:,2], color="purple", s=15)

     for i, (x,y,z) in enumerate(pts):
        ax.text(x, y, z, str(i+1), color="black", fontsize=8) 

     ax.view_init(elev=20)



     plt.show()


# --- Optimalizace ---

    @staticmethod
    def f(x, y, z):
        return ((x - 0.3)**2 + (y - 0.7)**2)**1/2 + z + 1

    def F(self, t, n):
        x, y, z = self.hilbert_polygon_point(t,n)
        return self.f(x, y, z)
   
    # --- Zabudovana python metoda pro hledani minima ---
   
    def find_minimum(self,n):
        result = minimize_scalar(lambda t: self.F(t,n), bounds=(0, 1), method='bounded')
        t_min = result.x
        h_min = self.hilbert_polygon_point(t_min,n)
        f_min = self.f(*h_min)
        return t_min, h_min, f_min
   
    # --- Optimalizacni algoritmus pro hledani minima ---
    def Holder_algorithm(self,H,r,eps,max_iter,n):
        N = 3                      
        # STEP 0: inicializace
        xk = [0.0, 1.0]
        zk = [self.F(0,n), self.F(1,n)]
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
        t_min = xk[min_idx]               
        x_min, y_min, z_min = self.hilbert_polygon_point(t_min,n)  
        f_min = self.f(x_min, y_min, z_min)      

        return t_min, f_min, x_min, y_min, z_min