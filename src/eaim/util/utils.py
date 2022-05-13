import random


def bounded(x, lb, ub):
    if x < lb:
        return lb
    elif x > ub:
        return ub
    return x


def random_coordinate(domain: list[float]) -> list[float]:
    """
    Generate random `d` dimension coordinates (Random Search Heuristic)

    domain = The domain for each dimension of the coordinates to be generated
    """
    return 


