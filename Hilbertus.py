import math
import numpy as np
from scipy.optimize import minimize_scalar

def dec_to_octal(Number,precision):   
    Q_Num = []
    i = 0
    if 0 < Number < 1:
     
     while Number != 0 and i < precision:
        Number = Number * 8
        digit = math.floor(Number)
        Q_Num.append(digit)
        Number = Number - digit
        i += 1

    return Q_Num


def dec_to_quarter(Number,precision):   
    Q_Num = []
    i = 0
    if 0 < Number < 1:
     
     while Number != 0 and i < precision:
        Number = Number * 4
        digit = math.floor(Number)
        Q_Num.append(digit)
        Number = Number - digit
        i += 1
   
    elif (Number == 1):
       Q_Num.append(1)
    
    elif (Number == 0):
       Q_Num.append(0)

    return Q_Num


def ej_and_dj_counter(Q_Num):
    e0j_counted = np.zeros(len(Q_Num))
    e3j_counted = np.zeros(len(Q_Num))
    dj_counted = np.zeros(len(Q_Num))
   
    for i in range(1,len(Q_Num)):
      for j in range(i):
        if Q_Num[j] == 3:
          e3j_counted[i] += 1
        elif Q_Num[j] == 0:
           e0j_counted[i] += 1
        else:
           pass 

    e3j_counted = e3j_counted % 2
    e0j_counted = e0j_counted % 2
    dj_counted = (e3j_counted + e0j_counted) % 2
    return e0j_counted, dj_counted


def calculate_point(e0j_counted, dj_counted, Q_Num):
   sum = np.zeros((2, 1))
   for i in range (1,len(Q_Num)+1):
      sum+= (1/(2**i))*((-1)**e0j_counted[i-1])*(np.sign(Q_Num[i-1])*np.array([[(1-dj_counted[i-1])*Q_Num[i-1]-1], [1-dj_counted[i-1]*Q_Num[i-1]]]))
   return sum

def hilbert_point(t,precision):
   Q = dec_to_quarter(t,precision)
   e0, dj = ej_and_dj_counter(Q)
   point = calculate_point(e0, dj, Q)
   return point.flatten()


def f(x,y):
   return (x - 0.5)**2 + (y - 0.5)**2

def F(t, precison):
   x, y = hilbert_point(t, precison)
   return f(x, y)

 #Brentova metoda na omezeném intervalu 
def find_minimum(precision):
    result = minimize_scalar(lambda t: F(t, precision), bounds=(0,1), method='bounded')
    t_min = result.x
    h_min = hilbert_point(t_min, precision)
    f_min = f(*h_min)
    return t_min, h_min, f_min

def ThreeD_Hilbert(Q_Num):

   H_0 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
   H_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
   H_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
   H_3 = np.array([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]])
   H_4 = np.array([[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]])
   H_5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
   H_6 = np.array([[ 0,  0, -1], [ 0,  1,  0], [-1,  0,  0]])
   H_7 = np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]])


   h_0 = np.array([0, 0, 0])
   h_1 = np.array([0, 1, 1])
   h_2 = np.array([1, 1, 0])
   h_3 = np.array([1, 1, 1])
   h_4 = np.array([2, 1, 1])
   h_5 = np.array([1, 1, 1])
   h_6 = np.array([1, 1, 2])
   h_7 = np.array([0, 1, 2])


   H_all = [H_0, H_1, H_2, H_3, H_4, H_5, H_6, H_7]
   h_all = [h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7]
   i = 1
   soucin = np.eye(3)
   sum = h_all[Q_Num[0]]*(1/2)
   
   for j in range(1,len(Q_Num)):
      for k in range(j):
        
         soucin = soucin @ H_all[Q_Num[k]]
         print(H_all[Q_Num[k]])
      sum += (1/(2**(j+1)))*(soucin @ h_all[Q_Num[j]])
      print(sum)
      print(h_all[Q_Num[j]])
      
      soucin = np.eye(3)

   return sum


L= dec_to_octal(0.2,15)
h = [2, 0, 6 ]
result = ThreeD_Hilbert(h)
print(result)


for p in [4, 6, 8, 10, 150]:
    precision = p
    t_min, h_min, f_min = find_minimum(precision)
    print(f"precision={p:2d}  =>  t = {t_min:.6f},  h(t) = ({h_min[0]:.4f}, {h_min[1]:.4f}),  f = {f_min:.6f}")




Q = dec_to_quarter(0.2,15)
print(Q)
e0, dj = ej_and_dj_counter(Q)
result = calculate_point(e0, dj, Q)
print(result)


h = [2, 0, 3]
e0, dj = ej_and_dj_counter(h)
print(e0)
print(dj)

result = calculate_point(e0, dj, h)
print(result)

