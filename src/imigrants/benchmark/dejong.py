import math
import numpy as np
from typing import Callable

# De Jong Functions
# https://www1.icsi.berkeley.edu/ftp/pub/techreports/1995/tr-95-012.pdf


def f1(x: list[float]) -> float:
    """
    De Jong F1 or the Sphere function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return sum(i**2 for i in x)


def f2(x: list[float]) -> float:
    """
    De Jong F2 or Rosenbrock function
    domain: [-2.048, 2.048] for each dimension.
    min = 0 at x = (1,1,...,1)
    """
    return sum(1 - x[i] +
               100 * (x[i + 1] - x[i]**2)**2
               for i in range(len(x) - 1))


def f3(x: list[float]) -> float:
    """
    De Jong F3 or Step function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (-5.12, -5.12,...,-5.12)
    """
    return 6*len(x) + sum(math.floor(i) for i in x)


def f4(x: list[float],
       noise: Callable = np.random.normal) -> float:
    """
    De Jong F4 or Quartic function
    domain: [-1.28, 1.28] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return sum(i * k**4 for i, k in enumerate(x)) + noise()


def f5(x: list[float]) -> float:
    """
    De Jong F5 or Shekel's Foxholes function
    domain: [-65.536, 65.536] for each dimension d in (1,2).
    min = 0.998 at x = (-32,-32)
    """

    def a(i, j):
        A = [-32.0, -16.0, 0.0, 16.0, 32.0]
        return A[i % 5] if j == 0 else A[(i // 5) % 5]

    return 1.0 / (0.002 +
                  sum(1.0 /
                      (i + sum((x[j] - a(i, j)) ** 6 for j in range(1))
                       for i in range(24))))
