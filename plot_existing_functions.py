"""
Vykreslení 3D grafů pro existující funkce v Hilbert2D
S vizualizací Hölderova optimalizačního algoritmu
"""
import numpy as np
from Hilbert2D import Hilbert2D
from Hilbert2DVisualizer import Hilbert2DVisualizer
from Hilbert2Dmainstream import Hilbert2Dmainstream
from Hilbert2DmainstreamVisualizer import Hilbert2DmainstreamVisualizer

# Vytvoření instancí - oddělená výpočetní a vizualizační třída
h2d = Hilbert2D(precision=20)
viz = Hilbert2DVisualizer(h2d)

print("="*60)
print("VYKRESLOVÁNÍ S HÖLDEROVÝM OPTIMALIZAČNÍM ALGORITMEM")
print("="*60)

# ============= PŘÍKLAD 1: Základní funkce f s optimalizací =============
print("\n1. Vykreslování základní funkce f(x,y) s optimalizací...")
fig, ax, t_min, f_min, n_points = viz.plot_function_with_hilbert_and_optimization(
    func=h2d.f,
    n=5,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1,  
    r=1.1,
    eps=1e-5,
    max_iter=100,
    grid_points=50,
    curve_samples=4**5,
    title="Základní funkce f(x,y) s Hölderovým algoritmem",
    true_min=1.0
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")


print("\n" + "="*60)
print("VYKRESLOVÁNÍ S MAINSTREAM HILBERTOVOU KŘIVKOU")
print("="*60)

# Vytvoření instancí pro mainstream variantu
h2d_main = Hilbert2Dmainstream(precision=20)
viz_main = Hilbert2DmainstreamVisualizer(precision=20)

# ============= MAINSTREAM PŘÍKLAD 1: Základní funkce f s optimalizací =============
print("\n4. [Mainstream] Vykreslování základní funkce f(x,y) s optimalizací...")
fig, ax, t_min, f_min, n_points = viz_main.plot_function_with_hilbert_and_optimization(
    func=h2d_main.f,
    n=5,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1,
    r=1.1,
    eps=1e-5,
    max_iter=100,
    grid_points=50,
    curve_samples=4**5,
    title="[Mainstream] Základní funkce f(x,y) s Hölderovým algoritmem",
    true_min=1.0
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")


# ============= MAINSTREAM PŘÍKLAD 2: Funkce f1_square s optimalizací =============
print("\n5. [Mainstream] Vykreslování f1_square s optimalizací...")
fig, ax, t_min, f_min, n_points = viz_main.plot_function_with_hilbert_and_optimization(
    func=h2d_main.f1_square,
    n=6,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1,
    r=4.1,
    eps=1e-6,
    max_iter=150,
    grid_points=60,
    curve_samples=4**6,
    title="[Mainstream] Funkce f1_square s Hölderovým algoritmem",
    true_min=-1.9133
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")


# ============= MAINSTREAM PŘÍKLAD 3: Matyas funkce f2_square s optimalizací =============
print("\n6. [Mainstream] Vykreslování Matyas funkce f2_square s optimalizací...")
fig, ax, t_min, f_min, n_points = viz_main.plot_function_with_hilbert_and_optimization(
    func=h2d_main.f2_square,
    n=10,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1,
    r=3,
    eps=1e-5,
    max_iter=350,
    grid_points=50,
    curve_samples=4**10,
    title="[Mainstream] Matyas funkce f2_square s Hölderovým algoritmem",
    true_min=0.0
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")


# ============= PŘÍKLAD 2: Funkce f1_square s optimalizací =============
print("\n2. Vykreslování f1_square s optimalizací...")
fig, ax, t_min, f_min, n_points = viz.plot_function_with_hilbert_and_optimization(
    func=h2d.f1_square,
    n=6,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1, 
    r=4.1,
    eps=1e-6,
    max_iter=150,
    grid_points=60,
    curve_samples=4**6,
    title="Funkce f1_square s Hölderovým algoritmem",
    true_min=-1.9133
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")


# ============= PŘÍKLAD 3: Matyas funkce f2_square s optimalizací =============
print("\n3. Vykreslování Matyas funkce f2_square s optimalizací...")
fig, ax, t_min, f_min, n_points = viz.plot_function_with_hilbert_and_optimization(
    func=h2d.f2_square,
    n=10,
    x_range=(0, 1),
    y_range=(0, 1),
    H=-1,  
    r=3,
    eps=1e-5,
    max_iter=350,
    grid_points=50,
    curve_samples=4**10,
    title="Matyas funkce f2_square s Hölderovým algoritmem",
    true_min=0.0
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")

