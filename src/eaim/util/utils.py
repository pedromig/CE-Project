
def bounded(x, lb, ub):
    if x < lb:
        return lb
    elif x > ub:
        return ub
    return x
