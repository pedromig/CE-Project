#!/usr/bin/env python3
import random

from eaim.ea import EvolutionaryAlgorithm

# Mutation Operators
from eaim.operators import (UniformBitflipMutation,
                            GaussianMutation,
                            UniformMutation)

# Recombination Operators
from eaim.operators import (NPointCrossover,
                            ArithmeticCrossover,
                            UniformCrossover)

# Parent Selection Operators
from eaim.operators import KTournamentSelection, RouletteWheelSelection

# Survivor Selection Strategy
from eaim.operators import Elitism

# Immigrant Insertion
from eaim.operators import RandomImmigrants, ElitistImmigrants

# João Brandão's Numbers Benchmark Probelm
from eaim.benchmark import JB

# Function Optimization Benchmark Problem
from eaim.benchmark import Function
from eaim.benchmark import sphere, rosenbrock, step, quartic
from eaim.benchmark import rastringin, schewefel, griewank


def test_jb(size):
    john = EvolutionaryAlgorithm(
        popsize=100,
        generations=1000,
        problem=JB,
        mutation=UniformBitflipMutation(probability=0.2),
        crossover=NPointCrossover(probability=0.2, points=1),
        selection=KTournamentSelection(size=100, k=2),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=UniformBitflipMutation(
                                         probability=0.2))
    )
    john(size, seed=42)

    # Stats -> (generation, best individual, average population fitness)
    stats = john.statistics()
    print(stats[0][1])
    print(stats[-1][1])

    # TODO: function that takes statistics and plots them over generation
    # fitnes of the best and average over all the generations

    # Function that runs the problem with multiple seeds and saves the
    # infomation of multiple runs in a file (in this case
    # john(size, seed=XXXXXX)). Its just a loop basically


def test_sphere(dim):
    population = 100
    d = [(-5.12, 5.12) for _ in range(dim)]

    # Random Sigmas for the GaussianMutation
    s = [random.random() for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=GaussianMutation(probability=0.2, sigma=s, domain=d),
        crossover=ArithmeticCrossover(probability=0.2, alpha=0.1),
        selection=RouletteWheelSelection(size=population),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, sphere, seed=123)


def test_rosenbrock(dim):
    population = 100
    d = [(-2.048, 2.048) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, rosenbrock)


def test_step(dim):
    population = 100
    d = [(-5.12, 5.12) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, step)


def test_quartic(dim):
    population = 100
    d = [(-1.28, 1.28) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, quartic)


def test_rastringin(dim):
    population = 100
    d = [(-5.12, 5.12) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, rastringin)


def test_schewefel(dim):
    population = 100
    d = [(-500, 500) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, schewefel)


def test_griewank(dim):
    population = 100
    d = [(-600, 600) for _ in range(dim)]

    function = EvolutionaryAlgorithm(
        popsize=population,
        generations=1000,
        problem=Function,
        mutation=UniformMutation(probability=0.2, domain=d),
        crossover=UniformCrossover(probability=0.2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(0.2),
        immigrants=RandomImmigrants(immigrants=0.2)
    )
    function(dim, d, griewank)


if __name__ == "__main__":
    test_jb(20)
    # test_sphere(2)
    # test_rosenbrock(2)
    # test_step(2)
    # test_quartic(2)
    # test_rastringin(2)
    # test_schewefel(2)
    # test_griewank(2)
