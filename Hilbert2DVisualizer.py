import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches


class Hilbert2DVisualizer:
   
    
    def __init__(self, hilbert2d):
     
        self.hilbert = hilbert2d
    
    
    def compare_iteration_counts(self, n, x_min, x_max, y_min, y_max, whatFunc, true_min, ftol=1e-6, maxiter=1000, H=-2, I=2, r=3, eps=1e-6, max_iter_holder=1000):
       

        
        results = {}
        
        # 1. Minimize Scalar (Brent's method)
        
        try:
            t_min, h_min, f_min, nfev = self.hilbert.find_minimum_mapped(
                n, x_min, x_max, y_min, y_max, whatFunc, true_min, ftol, maxiter
            )
            x_mapped, y_mapped = h_min
            results['minimize_scalar'] = {
                'iterations': nfev,
                'f_min': f_min,
                'x': x_mapped,
                'y': y_mapped,
                'success': True
            }
        except Exception as e:
            results['minimize_scalar'] = {
                'iterations': None,
                'f_min': None,
                'success': False,
                'error': str(e)
            }
        
        # 2. Differential Evolution
       
        try:
            f_min, x_mapped, y_mapped, generations, nfev = self.hilbert.differential_evolution_mapped(
                x_min, x_max, y_min, y_max, whatFunc, true_min, ftol, maxiter
            )
            results['differential_evolution'] = {
                'iterations': nfev,
                'generations': generations,
                'f_min': f_min,
                'x': x_mapped,
                'y': y_mapped,
                'success': True
            }
        except Exception as e:
            results['differential_evolution'] = {
                'iterations': None,
                'f_min': None,
                'success': False,
                'error': str(e)
            }
            
        
      
        try:
            t_min, f_min, x_mapped, y_mapped, usedH_arr = self.hilbert.Holder_algorithm_mapped(
                H, I, r, eps, max_iter_holder, n, whatFunc, true_min, ftol
            )
            
            holder_iterations = len(usedH_arr)
            results['holder'] = {
                'iterations': holder_iterations,
                'f_min': f_min,
                'x': x_mapped,
                'y': y_mapped,
                'success': True
            }
        except Exception as e:
            results['holder'] = {
                'iterations': None,
                'f_min': None,
                'success': False,
                'error': str(e)
            }
         
        
       
        
        return results


    def compare_iterations_by_curve_order(self, n_values, x_min, x_max, y_min, y_max, whatFunc, true_min, ftol=1e-6, maxiter=1000, H=-1, I=2, r=3, eps=1e-6, max_iter_holder=1000):
     
        import pandas as pd
        
    
        
        all_results = {}
        
        for n in n_values:

            
            results = self.compare_iteration_counts(
                n, x_min, x_max, y_min, y_max, whatFunc, true_min, 
                ftol, maxiter, H, I, r, eps, max_iter_holder
            )
            
            all_results[n] = results
       
        print("\n" + "="*70)
        print("SOUHRNNÁ TABULKA - POČET ITERACÍ PRO RŮZNÉ ŘÁDY KŘIVKY")
        print("="*70 + "\n")
        
        
        table_data = []
        for n in n_values:
            row = {'n': n}
            for method in ['minimize_scalar', 'differential_evolution', 'holder']:
                if all_results[n][method]['success']:
                    row[method] = all_results[n][method]['iterations']
                else:
                    row[method] = 'FAIL'
            table_data.append(row)
        
        # Vytvoření pandas DataFrame pro hezčí výpis
        df = pd.DataFrame(table_data)
        df.columns = ['n', 'Minimize Scalar', 'Diff. Evolution', 'Holder']
        print(df.to_string(index=False))
        
       
        
        for method in ['minimize_scalar', 'differential_evolution', 'holder']:
            method_name = method.replace('_', ' ').title()
            iterations = [all_results[n][method]['iterations'] 
                         for n in n_values 
                         if all_results[n][method]['success'] and all_results[n][method]['iterations'] is not None]
            
        

        
        return all_results, df











    def plot_hilbert_polygon(self, n):
        

        N = 2**(2*n)
        points = []

        for k in range(N+1):
            t = k / N
            points.append(self.hilbert.hilbert_polygon_point(t,n))

        points = np.array(points)
        plt.plot(points[:,0], points[:,1], '-o', markersize=2)
        plt.axis('equal')
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
                p = self.hilbert.hilbert_polygon_point(t, n)
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
          
            _, f_m, _, _, _ = self.hilbert.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, whatFunc, true_min, ftol=eps)
            diff_m = abs(f_m - true_min)

       
            _,_,f_nm,_ = self.hilbert.find_minimum_mapped(n, x_min, x_max, y_min, y_max, whatFunc,true_min, ftol=eps)
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

    def compare_holder_variants(self, r, eps, max_iter, N_vals, whatFunc, true_min):
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
                _, f_min, _, _, _ = self.hilbert.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, whatFunc, true_min, ftol=eps)
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

#UPRAVIT!!!!!!!!!!!!!!!!!!!!!!!!!!


    def compare_H_approximations_iterations(self, H_exact, r, eps, max_iter, n_vals, whatFunc, true_min, ftol, I=2):
       
        if not isinstance(ftol, list):
            ftol_list = [ftol]
        else:
            ftol_list = ftol
            
        results = []
        
        # Testování pro každou přesnost ftol
        for tol in ftol_list:
         
            
            for n in n_vals:
                iteration_counts = {}
                
              
               
                _, f_min_exact, _, _, usedH_exact = self.hilbert.Holder_algorithm_mapped(
                    H_exact, I, r, eps, max_iter, n, whatFunc, true_min, ftol=tol
                )
                iter_exact = len(usedH_exact)
                error_exact = abs(f_min_exact - true_min)
                iteration_counts['H_exact'] = iter_exact
              
                
           
                _, f_min_h1, _, _, usedH_h1 = self.hilbert.Holder_algorithm_mapped(
                    -1, I, r, eps, max_iter, n, whatFunc, true_min, ftol=tol
                )
                iter_h1 = len(usedH_h1)
                error_h1 = abs(f_min_h1 - true_min)
                iteration_counts['H=-1'] = iter_h1
             
                
            
                _, f_min_h2, _, _, usedH_h2 = self.hilbert.Holder_algorithm_mapped(
                    -2, I, r, eps, max_iter, n, whatFunc, true_min, ftol=tol
                )
                iter_h2 = len(usedH_h2)
                error_h2 = abs(f_min_h2 - true_min)
                iteration_counts['H=-2'] = iter_h2
       
                
                # Uložení výsledků
                results.append({
                    'ftol': tol,
                    'n': n,
                    'H_exact_iter': iter_exact,
                    'H_exact_error': error_exact,
                    'H=-1_iter': iter_h1,
                    'H=-1_error': error_h1,
                    'H=-2_iter': iter_h2,
                    'H=-2_error': error_h2
                })
        
        # Vytvoření DataFrame
        df = pd.DataFrame(results)
        
        # Vykreslení grafů
        self._plot_H_comparison(df, n_vals, ftol_list)
        
        return df
    
    
    def _plot_H_comparison(self, df, n_vals, ftol_list):
        """
        Pomocná funkce pro vykreslení výsledků porovnání H aproximací.
        """
        num_ftol = len(ftol_list)
        
        # Graf 1: Počet iterací vs n pro různé H
        fig1, axes1 = plt.subplots(1, num_ftol, figsize=(8*num_ftol, 6.5))
        if num_ftol == 1:
            axes1 = [axes1]
        
        for idx, tol in enumerate(ftol_list):
            ax = axes1[idx]
            df_tol = df[df['ftol'] == tol]
            
            n_arr = df_tol['n'].to_numpy()
            h_exact_iter = df_tol['H_exact_iter'].to_numpy()
            h1_iter = df_tol['H=-1_iter'].to_numpy()
            h2_iter = df_tol['H=-2_iter'].to_numpy()
            
            ax.plot(n_arr, h_exact_iter, 'o-', label=f'H přesné', linewidth=2, markersize=8)
            ax.plot(n_arr, h1_iter, 's-', label='H=-1 (aprox 1)', linewidth=2, markersize=8)
            ax.plot(n_arr, h2_iter, '^-', label='H=-2 (aprox 2)', linewidth=2, markersize=8)
            
            ax.set_xlabel('Řád Hilbertovy křivky (n)', fontsize=12)
            ax.set_ylabel('Počet iterací', fontsize=12)
            ax.set_title(f'Počet iterací pro ftol={tol}', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Graf 2: Chyba vs n pro různé H
        fig2, axes2 = plt.subplots(1, num_ftol, figsize=(8*num_ftol, 6.5))
        if num_ftol == 1:
            axes2 = [axes2]
        
        for idx, tol in enumerate(ftol_list):
            ax = axes2[idx]
            df_tol = df[df['ftol'] == tol]
            
            n_arr = df_tol['n'].to_numpy()
            h_exact_error = df_tol['H_exact_error'].to_numpy()
            h1_error = df_tol['H=-1_error'].to_numpy()
            h2_error = df_tol['H=-2_error'].to_numpy()
            
            ax.plot(n_arr, h_exact_error, 'o-', label='H přesné', linewidth=2, markersize=8)
            ax.plot(n_arr, h1_error, 's-', label='H=-1 (aprox 1)', linewidth=2, markersize=8)
            ax.plot(n_arr, h2_error, '^-', label='H=-2 (aprox 2)', linewidth=2, markersize=8)
            
            ax.set_xlabel('Řád Hilbertovy křivky (n)', fontsize=12)
            ax.set_ylabel('Chyba |f_min - true_min|', fontsize=12)
            ax.set_title(f'Chyba pro ftol={tol}', fontsize=13)
            ax.set_yscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, which='both')
            ax.axhline(y=tol, color='r', linestyle='--', alpha=0.5, label=f'ftol={tol}')
        
        plt.tight_layout()
        plt.show()
        

    def hyperparameter_tuning_r(self, r_values, H, I, eps, max_iter, N_vals, whatFunc, true_min):
        """
        Hyperparameter tuning pro parametr r
        """
        results = {f"r={r}": [] for r in r_values}
        n_values = []
        
        for n in N_vals:
            n_values.append(n)
            for r in r_values:
                _, f_min, _, _, _ = self.hilbert.Holder_algorithm_mapped(H, I, r, eps, max_iter, n, whatFunc, true_min, ftol=eps)
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
        






    def analyze_holder_constants(self, H_true,I,H, r, eps, max_iter, N_vals, whatFunc):
    
        results = []
        
        for n in N_vals:
            _, _, _, _, usedH_arr = self.hilbert.Holder_algorithm_mapped(H,I, r, eps, max_iter, n, whatFunc, true_min=None, ftol=eps)
            
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
        zk = [self.hilbert.F_mapped(0.0, n, x_min, x_max, y_min, y_max, whatFunc), 
              self.hilbert.F_mapped(1.0, n, x_min, x_max, y_min, y_max, whatFunc)]
        
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
                h_used = H
                
            
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
            zk.append(self.hilbert.F_mapped(y_star, n, x_min, x_max, y_min, y_max, whatFunc))
            current_iteration += 1


 

    
    
    def _plot_paraboloids_at_iteration(self, xk, zk, h_used, r, N, n, x_min, x_max, y_min, y_max, whatFunc, iteration):
     
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Levý graf: 1D průběh funkce na Hilbertově křivce + paraboloidy
        
        P = 2**(2*n)
        curve_points = []
        t_values = []
        for k in range(P):
            t = k / P
            t_values.append(t)
            curve_points.append(self.hilbert.map_to_area(t, n, x_min, x_max, y_min, y_max))
        curve_points = np.array(curve_points)
        f_dense = [self.hilbert.F_mapped(t, n, x_min, x_max, y_min, y_max, whatFunc) for t in t_values]
        
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
                z_intersect = self.hilbert.F_mapped(y_intersect, n, x_min, x_max, y_min, y_max, whatFunc)
                
            
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
        points_2d = [self.hilbert.map_to_area(t, n, x_min, x_max, y_min, y_max) for t in xk]
        points_2d = np.array(points_2d)
        
        ax2.scatter(points_2d[:, 0], points_2d[:, 1], color='red', s=50, zorder=5)
        for i, (x, y) in enumerate(points_2d):
            ax2.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Hilbertova křivka
       
        curve_points = []

        for k in range(P):
            t = k / P
            curve_points.append(self.hilbert.map_to_area(t, n, x_min, x_max, y_min, y_max))

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



#################################################################
# ----------- OBECNÁ FUNKCE PRO VYKRESLENÍ 3D GRAFU ------------#
#################################################################

    def plot_function_with_hilbert_curve(self, func, n, x_range=(-1, 1), y_range=(-1, 1), 
                                         grid_points=50, curve_samples=1000, 
                                         title="Funkce s Hilbertovou křivkou"):
   
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Vytvoření mřížky pro povrch
        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Výpočet hodnot funkce
        Z = np.zeros_like(X)
        for i in range(grid_points):
            for j in range(grid_points):
                Z[i, j] = func(X[i, j], Y[i, j])
        
        # Vytvoření 3D grafu
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vykreslení povrchu
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0.5, edgecolor='black', antialiased=True)
        
        curve_points = []
        for k in range(curve_samples):
            t = k / curve_samples
            # Získání bodu na jednotkové křivce [0,1]x[0,1]
            point = self.hilbert.hilbert_polygon_point(t, n)
            px, py = point
            # Mapování do požadovaného rozsahu
            new_x = x_min + (x_max - x_min) * px
            new_y = y_min + (y_max - y_min) * py
            # Z-ová souřadnice na minimální hodnotě (v základně)
            curve_points.append([new_x, new_y])
        
        curve_points = np.array(curve_points)
        
        # Nastavení z-souřadnice křivky na minimum grafu
        z_min = np.min(Z)
        z_curve = np.full(len(curve_points), z_min)
        
        # Vykreslení Hilbertovy křivky v základně
        ax.plot(curve_points[:, 0], curve_points[:, 1], z_curve, 
               'r-', linewidth=2, label='Hilbertova křivka', alpha=0.9)
        
        # Nastavení os a popisků
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('f(x, y)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Přidání colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Nastavení úhlu pohledu
        ax.view_init(elev=25, azim=45)
        
        # Legenda
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax




    def plot_function_with_hilbert_and_optimization(self, func, n, x_range=(-1, 1), y_range=(-1, 1),
                                                     H=2.0, I=2, r=1.1, eps=1e-5, max_iter=100,
                                                     grid_points=50, curve_samples=1000,
                                                     title="Funkce s Hilbertovou křivkou a optimalizací",
                                                     true_min=None, ftol=1e-5):
    
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Vytvoření mřížky pro povrch
        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Výpočet hodnot funkce
        Z = np.zeros_like(X)
        for i in range(grid_points):
            for j in range(grid_points):
                Z[i, j] = func(X[i, j], Y[i, j])
        
        # Wrapper pro algoritmus - vytvoříme dočasnou funkci kompatibilní s F()
        # Uložíme si historii pomocí closure
        t_history = []
        
        original_F = self.hilbert.F
        def custom_F(t, n_param, whatFunc_param):
            t_history.append(t)
            point = self.hilbert.hilbert_polygon_point(t, n)
            px, py = point
            return func(px, py)
        
        # Dočasně nahradíme self.hilbert.F
        self.hilbert.F = custom_F
        
        # Spustíme existující Hölderův algoritmus
        t_min, f_min, x_mapped, y_mapped, usedH_arr = self.hilbert.Holder_algorithm_mapped(
            H=H, I=I, r=r, eps=eps, max_iter=max_iter, n=n,
            whatFunc=0, true_min=true_min, ftol=ftol
        )
        
        # Obnovíme původní F
        self.hilbert.F = original_F
        
        # Vytvoření 3D grafu
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vykreslení povrchu
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, 
                              linewidth=0.3, edgecolor='black', antialiased=True)
        
        
        curve_points = []
        for k_idx in range(curve_samples):
            t = k_idx / curve_samples
            point = self.hilbert.hilbert_polygon_point(t, n)
            px, py = point
            curve_points.append([px, py])
        
        curve_points = np.array(curve_points)
        z_min_surface = np.min(Z)
        z_curve = np.full(len(curve_points), z_min_surface)
        
        # Vykreslení Hilbertovy křivky v základně
        ax.plot(curve_points[:, 0], curve_points[:, 1], z_curve, 
               'purple', linewidth=2, label='Hilbertova křivka', alpha=0.8)
        
        # Vykreslení testovaných bodů z algoritmu
        opt_points_2d = []
        opt_points_z = []
        
        for t in t_history:
            point = self.hilbert.hilbert_polygon_point(t, n)
            px, py = point
            z_val = func(px, py)
            opt_points_2d.append([px, py])
            opt_points_z.append(z_val)
        
        opt_points_2d = np.array(opt_points_2d)
        opt_points_z = np.array(opt_points_z)
        
        # Vykreslení bodů na křivce v základně
        z_points_base = np.full(len(opt_points_2d), z_min_surface)
        ax.scatter(opt_points_2d[:, 0], opt_points_2d[:, 1], z_points_base,
                  c='cyan', s=50, marker='o', label='Testované body (křivka)', 
                  edgecolors='black', linewidths=1, alpha=0.9)
        
        # Vykreslení bodů na povrchu funkce
        ax.scatter(opt_points_2d[:, 0], opt_points_2d[:, 1], opt_points_z,
                  c='red', s=50, marker='o', label='Testované body (funkce)',
                  edgecolors='black', linewidths=1, alpha=0.9)
        
        # Spojnice mezi body na křivce a na povrchu
        for i in range(len(opt_points_2d)):
            ax.plot([opt_points_2d[i, 0], opt_points_2d[i, 0]],
                   [opt_points_2d[i, 1], opt_points_2d[i, 1]],
                   [z_points_base[i], opt_points_z[i]],
                   'orange', linewidth=1.5, alpha=0.6)
        
        # Zvýraznění nalezeného minima
        point_min = self.hilbert.hilbert_polygon_point(t_min, n)
        px_min, py_min = point_min
        z_min_val = func(px_min, py_min)
        
        ax.scatter([px_min], [py_min], [z_min_val],
                  c='lime', s=200, marker='*', label='Nalezené minimum',
                  edgecolors='black', linewidths=2, zorder=5)
        
        ax.plot([px_min, px_min], [py_min, py_min],
               [z_min_surface, z_min_val],
               'lime', linewidth=3, alpha=0.9)
        
        # Nastavení os a popisků
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('f(x, y)', fontsize=12)
        ax.set_title(f"{title}\n(Testováno bodů: {len(t_history)}, Minimum: f={z_min_val:.6f})", 
                    fontsize=14, pad=20)
        
        # Přidání colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Nastavení úhlu pohledu
        ax.view_init(elev=25, azim=45)
        
        # Legenda
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, t_min, z_min_val, len(t_history)
