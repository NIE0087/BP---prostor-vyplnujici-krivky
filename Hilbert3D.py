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
    

    def ThreeD_Hilbert_point(self, t, n):
        
        
        q = self.dec_to_octal(t)
        point = self.ThreeD_Hilbert(q)
        return point.flatten()


    def hilbert_polygon_point(self, t, n):
    
        N = 2**(2*n)
    
    
        k = int(np.floor(t * N))
        #if t == 1.0: chaby pokus 
        #   t = 1.0 - 1e-12

        p_k = self.ThreeD_Hilbert_point(k / N)
        p_k1 = self.ThreeD_Hilbert_point((k + 1) / N)
        
        point = N * (t-(k/N))*p_k1 - N*(t-((k+1)/N))*p_k

        return point
   
    def plot_hilbert_polygon(self,n):
        N = 8**n
        pts = np.zeros((N, 3))

        for k in range(N):
            t= k/N
            pts[k] = self.ThreeD_Hilbert_point(t,n)  

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