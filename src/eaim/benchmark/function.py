import math
import random
import numpy as np

from typing import Callable


class Function:
    def __init__(self: object, dim: int,
                 domain: list[tuple[float, float]],
                 fn: Callable) -> None:
        self.genotype = [random.uniform(*domain[i]) for i in range(dim)]
        self.domain = domain
        self.function = fn
        self.eval()

    def __getitem__(self: object, key: int) -> int:
        return self.genotype[key]

    def __setitem__(self: object, key: int, value: int):
        self.genotype[key] = value

    def __len__(self: object) -> int:
        return len(self.genotype)

    def __lt__(self: object, other: object) -> bool:
        return self.fitness > other.fitness

    def eval(self: object) -> float:
        self.fitness = self.function(self.genotype)
        return self.fitness

    def __repr__(self: object):
        return str(self.genotype)

    def __str__(self: object):
        return f"({self.genotype}, {self.fitness})"


def sphere(x: list[float]) -> float:
    """
    De Jong F1 or the Sphere function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return sum(i**2 for i in x)


def rosenbrock(x: list[float]) -> float:
    """
    De Jong F2 or Rosenbrock function
    domain: [-2.048, 2.048] for each dimension.
    min = 0 at x = (1,1,...,1)
    """
    return sum((1 - x[i])**2 + 100 * (x[i + 1] - x[i]**2)**2
               for i in range(len(x) - 1))


def step(x: list[float]) -> float:
    """
    De Jong F3 or Step function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (-5.12, -5.12,...,-5.12)
    """
    return 6*len(x) + sum(math.floor(i) for i in x)


def quartic(x: list[float],
            N: Callable = np.random.normal) -> float:
    """
    De Jong F4 or Quartic function
    domain: [-1.28, 1.28] for each dimension.
    min = 0 at x = (0,0,...,0) with no noise
    """
    return sum(i * k**4 for i, k in enumerate(x, start=1)) + N(0, 1)


def rastringin(x: list[float], a: int = 10) -> float:
    """
    Rastringin function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return a*len(x) + sum(k**2 - a * math.cos(2 * math.pi * k) for k in x)


def schewefel(x: list[float]) -> float:
    """
    Schewefel function
    domain: [-500, 500] for each dimension.
    min = 0 at x = (420.9687,420.9687,...,420.9687)
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    return (418.9829 * len(x))  \
        - sum(k * math.sin(math.sqrt(abs(k))) for k in x)


def griewank(x: list[float]) -> float:
    """
    Griewank function
    domain: [-600, 600] for each dimension.
    min = 0 at x = (0,0,...,0)
    https://www.sfu.ca/~ssurjano/griewank.html
    """
    return 1 + (sum(k**2 for k in x) / 4000) \
        - np.prod(list(map(lambda t: math.cos(t[1] / math.sqrt(t[0])),
                  enumerate(x, start=1))))
