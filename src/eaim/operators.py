import random
import copy

from typing import Tuple, Callable
from .util import bounded


class UniformBitflipMutation:
    def __init__(self: object, probability: float) -> None:
        self._probability = probability

    def __call__(self: object, x) -> None:
        for i in range(len(x)):
            if random.random() < self._probability:
                x[i] ^= 1


class GaussianMutation:
    def __init__(self: object,
                 probability: float,
                 sigma: list[float],
                 domain: list[Tuple[float, float]]) -> None:
        self._probability = probability
        self._domain = domain
        self._sigma = sigma

    def __call__(self: object, x) -> None:
        for i in range(len(x)):
            if random.random() < self._probability:
                sol = x[i] + random.gauss(0, self._sigma[i])
                x[i] = bounded(sol, *self._domain[i])


class UniformMutation:
    def __init__(self: object,
                 probability: float,
                 domain: list[Tuple[float, float]]) -> None:
        self._probability = probability
        self._domain = domain

    def __call__(self: object, x) -> None:
        for i in range(len(x)):
            if random.random() < self._probability:
                x[i] = random.uniform(*self._domain[i])


class NPointCrossover:
    def __init__(self: object,
                 probability: float,
                 points: int = 1) -> None:
        self._probability = probability
        self._points = points

    def __call__(self: object, a, b) -> None:
        if random.random() < self._probability:
            points = sorted(random.sample(range(len(a)), self._points + 1))
            for i in range(len(points) - 1):
                l, h = points[i], points[i + 1]
                if i % 2 == 0:
                    a[l:h], b[l:h] = b[l:h], a[l:h]


class UniformCrossover:
    def __init__(self: object, probability: float) -> None:
        self._probability = probability

    def __call__(self: object, a, b) -> None:
        if random.random() < self._probability:
            for i in range(len(a)):
                if random.random() < 0.5:
                    a[i], b[i] = b[i], a[i]


class ArithmeticCrossover:
    def __init__(self: object, probability: float, alpha: float) -> None:
        self._probability = probability
        self._alpha = alpha

    def __call__(self: object, a, b) -> None:
        if random.random() < self._probability:
            for i in range(len(a)):
                x, y = a[i], b[i]
                a[i] = self._alpha * x + (1 - self._alpha) * y
                b[i] = self._alpha * y + (1 - self._alpha) * x


class KTournamentSelection:
    def __init__(self: object, size: int, k: int = 2) -> None:
        self._k = k
        self._size = size

    def __call__(self: object, population: list) -> list:
        matting_pool = []
        for i in range(self._size):
            tournament = random.sample(population, self._k)
            matting_pool.append(copy.deepcopy(max(tournament)))
        return matting_pool


class RouletteWheelSelection:
    def __init__(self: object, size: int,
                 fitness: Callable = lambda x: x.fitness):
        self._size = size
        self._fitness = fitness

    def __call__(self: object, population: list) -> None:
        fitnesses = [self._fitness(x) for x in population]
        total = sum(fitnesses)

        freq = [f / float(total) for f in fitnesses]
        roulette = [sum(freq[:i + 1]) for i in range(len(freq))]

        matting_pool = []
        for _ in range(self._size):
            value = random.uniform(0, 1)
            for i, s in enumerate(population):
                if value <= roulette[i]:
                    matting_pool.append(copy.deepcopy(s))
                    break
        return matting_pool


class Elitism:
    def __init__(self: object, elite: float) -> None:
        self._elite = elite

    def __call__(self: object, parents: list, offspring: list) -> None:
        e = int(len(parents) * self._elite)
        offspring.sort(reverse=True)
        parents.sort(reverse=True)
        return parents[:e] + offspring[:len(parents) - e]


class RandomImmigrants:
    def __init__(self: object, immigrants: float) -> None:
        self._immigrants = immigrants

    def __call__(self: object, population: list,
                 problem, *args, **kwargs) -> None:
        immigrants = int(len(population) * self._immigrants)
        population.sort()
        for i in range(immigrants):
            population[i] = problem(*args, **kwargs)


class ElitistImmigrants:
    def __init__(self: object, immigrants: float, mutation: Callable) -> None:
        self._immigrants = immigrants
        self._mutation = mutation

    def __call__(self: object, population: list, *args, **kwargs) -> None:
        immigrants = int(len(population) * self._immigrants)
        population.sort()
        for i in range(immigrants):
            population[i] = copy.deepcopy(population[-1])
            self._mutation(population[i])
