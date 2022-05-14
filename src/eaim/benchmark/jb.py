import random


class JB:
    """
    João Brandão's Numbers Problem

    alpha = Weight of the reward associated with the number of
            correct choices of numbers for the final set
    beta = Weight of the penalty associated with the number of
           incorrect choices of numbers (that violate the problem
           constraints) for the final set
    """

    def __init__(self: object, size: int = 0,
                 alpha: float = 1.0, beta: float = 1.5) -> None:
        self.genotype = [random.choice((0, 1)) for _ in range(size)]
        self.alpha = 1.0
        self.beta = 1.5
        self.eval()

    def eval(self: object) -> float:
        violations = 0
        for i in range(1, len(self.genotype)):
            v, lim = 0, min(i, len(self.genotype) - i - 1)
            for j in range(1, lim + 1):
                if (self.genotype[i] == 1) and (self.genotype[i - j] == 1) \
                        and (self.genotype[i + j] == 1):
                    v += 1
            violations += v
        self.fitness = self.alpha * sum(self.genotype) - self.beta * violations
        return self.fitness

    def __getitem__(self: object, key: int) -> int:
        return self.genotype[key]

    def __setitem__(self: object, key: int, value: int):
        self.genotype[key] = value

    def __len__(self: object) -> int:
        return len(self.genotype)

    def __lt__(self: object, other: object) -> bool:
        return self.fitness < other.fitness

    def __repr__(self: object) -> str:
        return str(self.genotype)

    def __str__(self: object) -> str:
        #fenotype = [n for n, g in enumerate(self.genotype) if g == 1]
        return f"{self.genotype}"