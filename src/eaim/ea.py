import random
from typing import Callable


class EvolutionaryAlgorithm:
    def __init__(self: object, *args, popsize: int, generations: int,
                 problem: Callable, selection: Callable,
                 crossover: Callable, mutation: Callable,
                 survivors: Callable,
                 immigrants: Callable = lambda *args, **kwags: None,
                 fitness: Callable = lambda x: x.fitness,
                 eval: Callable = lambda x: x.eval(),
                 **kwargs):

        self._problem = problem

        self._fitness = fitness
        self._eval = eval

        self._popsize = popsize
        self._generations = generations

        self._mutation = mutation
        self._crossover = crossover
        self._selection = selection

        self._survivors = survivors
        self._immigrants = immigrants

        self._args = args
        self._kwargs = kwargs

        self._stats = []

    def __call__(self: object, *args, stats=True, seed=None, **kwargs):

        # Set seed for the rng on this run
        random.seed(seed)

        # Generate initial population
        population = [self._problem(*args, **kwargs)
                      for i in range(self._popsize)]

        # Gather statistics - Generation, Curr Best, Average best (start)
        if stats:
            self._stats.append((0,
                                max(population),
                                sum(self._fitness(i)
                                    for i in population) / len(population)))

        for gen in range(1, self._generations + 1):
            # Parent Selection
            matting_pool = self._selection(
                population, *self._args, **self._kwargs)

            # Recombination
            for i in range(0, len(matting_pool) - 1, 2):
                a, b = matting_pool[i], matting_pool[i + 1]
                self._crossover(a, b, *self._args, **self._kwargs)

            # Mutation
            for solution in matting_pool:
                self._mutation(solution, *self._args, **self._kwargs)
                self._eval(solution)

            # Survivor Selection
            population = self._survivors(
                population, matting_pool, *self._args, **self._kwargs)

            # Gather statistics - Generation, Curr Best, Average best
            if stats:
                print(f"Generation {gen} of {self._generations}", end="\r")
                data = (gen,
                        max(population),
                        sum(self._fitness(i)
                            for i in population) / len(population))
                self._stats.append(data)

            # Immigrants Insertion
            if gen != self._generations:
                self._immigrants(population, self._problem, *args, **kwargs)

        return population

    def statistics(self: object):
        return self._stats
