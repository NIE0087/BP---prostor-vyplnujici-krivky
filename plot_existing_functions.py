
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
    max_iter=10,
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
    title=" Funkce f_0(x,y)",
    true_min=1.0
)
print(f"    Minimum: f = {f_min:.8f} při t = {t_min:.6f}")
print(f"    Počet testovaných bodů: {n_points}")











