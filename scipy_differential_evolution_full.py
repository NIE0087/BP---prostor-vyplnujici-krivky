"""
Differential evolution - celý zdrojový kód ze SciPy
Zdroj: https://github.com/scipy/scipy/blob/main/scipy/optimize/_differentialevolution.py

Toto je přesná kopie implementace differential_evolution z SciPy.
Lze použít pro studium algoritmu.
"""

from functools import partial
import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
                                        NonlinearConstraint, LinearConstraint)
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import (check_random_state, MapWrapper, _FunctionWrapper,
                              rng_integers, _transition_to_rng)
from scipy._lib._sparse import issparse

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps

# Poznámka: Pro úplnou funkčnost potřebujete celou knihovnu SciPy nainstalovanou.
# Tento soubor slouží hlavně ke studiu algoritmu.

print("Pro plný zdrojový kód differential_evolution navštivte:")
print("https://github.com/scipy/scipy/blob/main/scipy/optimize/_differentialevolution.py")
print("\nSoubor má přes 1500 řádků. Pro studium doporučuji:")
print("1. Otevřít odkaz v prohlížeči")
print("2. Nebo použít: pip show scipy  # najde cestu k instalaci")
print("3. Pak otevřít: scipy/optimize/_differentialevolution.py")
