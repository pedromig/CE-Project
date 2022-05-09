from typing import Tuple, List

# Solution type expected by the "João Brandão" fenotype and fitness functions
JBSolution = Tuple[List[int], float]


# João Brandão Problem (Benchmark)
def jb_fitness(solution: JBSolution,
               alpha: float = 1.0,
               beta: float = 1.5) -> float:
    """
    João Brandão's Numbers Problem - Fitness Function

    alpha = Weight of the reward associated with the number of
            correct choices of numbers for the final set
    beta = Weight of the penalty associated with the number of
           incorrect choices of numbers (that violate the problem
           constraints) for the final set
    """
    violations = 0
    for i in range(1, len(solution)):
        v, lim = 0, min(i, len(solution) - i - 1)
        for j in range(1, lim + 1):
            if (solution[i] == 1) and (solution[i - j] == 1) \
                    and (solution[i + j] == 1):
                v += 1
        violations += v
    return alpha * sum(solution) - beta * violations


def jb_fenotype(solution: JBSolution) -> list[int]:
    """
    João Brandão's Numbers Problem - Fenotype Translation Function

    solution = The solution (set representation) associated with the
               genotype provided in the `solution` supplied as the
               function parameter
    """
    return [n for n, g in enumerate(solution) if g == 1]
