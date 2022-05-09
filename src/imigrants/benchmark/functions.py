import math
from functools import reduce

# Rastringin Function
# https://www.sfu.ca/~ssurjano/rastr.html


def rastringin(x: list[float]) -> float:
    """
    Rastringin function
    domain: [-5.12, 5.12] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return 10*len(x) + sum(k**2 - 10 * math.cos(2 * math.pi * k) for k in x)


# Schewefel Function
# https://www.sfu.ca/~ssurjano/schwef.html
def schewefel(x: list[float]) -> float:
    """
    Schewefel function
    domain: [-500, 500] for each dimension.
    min = 0 at x = (420.9687,420.9687,...,420.9687)
    """
    return 418.9829 * len(x) - sum(k * math.sin(math.sqrt(abs(k))) for k in x)


# Griewank Function
# https://www.sfu.ca/~ssurjano/griewank.html
def griewank(x: list[float]) -> float:
    """
    Griewank function
    domain: [-600, 600] for each dimension.
    min = 0 at x = (0,0,...,0)
    """
    return sum((k**2 / 4000) - sum(map(
        lambda t: math.cos(t[0] / math.sqrt(t[1])), enumerate(x, start=1)
    )) for k in x) + 1


def trap(x: list[int]) -> float:
    """
    Trap Function
    domain : {0, 1} for each dimension
    max: len(x == 0) at x = (0,0,...,0)
    """
    return (len(x) - sum(x)) + ((len(x) + 1) * reduce(lambda x, y: x * y, x))
