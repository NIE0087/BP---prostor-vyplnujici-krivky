import math
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Hilbert3D:
    
    def __init__(self, precision: int):
        self.precision = precision
        self.heat_problem = None
        self.A_min = 1.0
        self.A_max = 10.0

    def configure_heat_problem(
        self,
        grid_size=50,
        sigma=0.08,
        noise_level=0.01,
        true_params=(0.35, 0.70, 5.0),
        random_seed=123,
        A_min=1.0,
        A_max=10.0,
    ):
        """Nastavi 3D inverzni problem zdroje tepla pro whatFunc=2."""
        import importlib
        import HeatSourceProblem

        HeatSourceProblem = importlib.reload(HeatSourceProblem)

        heat_cls = getattr(HeatSourceProblem, "HeatSourceInverseProblem3D", None)
        if heat_cls is None:
            heat_cls = getattr(HeatSourceProblem, "HeatSourceInverseProblem", None)
        if heat_cls is None:
            raise AttributeError(
                "HeatSourceProblem module nema tridu HeatSourceInverseProblem3D ani HeatSourceInverseProblem."
            )

        self.heat_problem = heat_cls(
            grid_size=grid_size,
            sigma=sigma,
            noise_level=noise_level,
            true_params=true_params,
            random_seed=random_seed,
        )
        self.A_min = float(A_min)
        self.A_max = float(A_max)

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

    def f2(self, x, y, z):
        """Cenova funkce inverzni ulohy: J(x,y,A), kde A je mapovane ze z."""
        if self.heat_problem is None:
            self.configure_heat_problem()

        a_unit = float(np.clip(z, 0.0, 1.0))
        A_current = self.A_min + a_unit * (self.A_max - self.A_min)
        return self.heat_problem.cost((x, y, A_current))

    def f2_cube(self, x, y, z):
        """Zpetna kompatibilita: pro whatFunc=2 vraci stejnou hodnotu jako f2."""
        return self.f2(x, y, z)

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
   
    
    def Holder_algorithm(self, H, r, eps, max_iter, n, true_min=None, ftol=1e-6, stop_condition="eps", I=1, whatFunc=0, return_iterations=False):
        """Spustí Holderův 1D algoritmus nad Hilbertovsky mapovanou funkcí."""
        N = 3                      
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
            actual_iterations = iteracni_krok + 1
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