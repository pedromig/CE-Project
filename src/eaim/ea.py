from typing import Callable


class EvolutionaryAlgorithm:
    def __init__(self: object, *args, popsize: int, generations: int,
                 problem: Callable, selection: Callable,
                 crossover: Callable, mutation: Callable,
                 survivors: Callable,
                 immigrants: Callable = lambda *args, **kwags: None,
                 **kwargs):

        self._problem = problem

        self._popsize = popsize
        self._generations = generations

        self._mutation = mutation
        self._crossover = crossover
        self._selection = selection

        self._survivors = survivors
        self._immigrants = immigrants

        self._args = args
        self._kwargs = kwargs

    def __call__(self: object, *args, **kwargs):
        population = [self._problem(*args, **kwargs)
                      for i in range(self._popsize)]

        print(max(population))
        for gen in range(1, self._generations + 1):
            matting_pool = self._selection(
                population, *self._args, **self._kwargs)

            for i in range(0, len(matting_pool) - 1, 2):
                a, b = matting_pool[i], matting_pool[i + 1]
                self._crossover(a, b, *self._args, **self._kwargs)

            for solution in matting_pool:
                self._mutation(solution, *self._args, **self._kwargs)
                solution.eval()

            population = self._survivors(
                population, matting_pool, *self._args, **self._kwargs)

            # Gather statistics - Curr Best, Average best, etc...
            print(f"Generation {gen} of {self._generations}", end="\r")

            # Immigrants Insertion
            self._immigrants(population, self._problem, *args, **kwargs)

        print(max(population))
        return population

    def stats():
        ...
