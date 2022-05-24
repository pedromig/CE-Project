#!/usr/bin/env python3
import random
# import matplotlib.pyplot as plt
import os

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

# João Brandão's Numbers Benchmark Problem
from eaim.benchmark import JB

# Function Optimization Benchmark Problem
from eaim.benchmark import Function
from eaim.benchmark import sphere, rosenbrock, step, quartic


def run(default, rand, elitist, runs, file, *args, **kwargs):
    seeds = random.sample(range(1, 1000), runs)
    statistics = []
    print("Running Experiments: ")
    for i, s in enumerate(seeds):
        print(f">> Running with seed {i} of {runs}")

        # Run Algorithm Versions
        default(*args, **kwargs, seed=s)
        rand(*args, **kwargs, seed=s)
        elitist(*args, **kwargs, seed=s)

        # a = [x[2] for x in default.statistics()]
        # b = [x[2] for x in rand.statistics()]
        # c = [x[2] for x in elitist.statistics()]

        # gens = list(range(len(a)))

        # plt.plot(gens, a, label="Evolutionary Algorithm")
        # plt.plot(gens, b, label="Random Immigrants")
        # plt.plot(gens, c, label="Elitist Immigrants")

        # axes = plt.gca()
        # axes.set_ylim((-1, 2))
        # plt.legend()
        # plt.savefig(f"test-{s}.pdf")

        # Record statistics
        statistics.append((default.statistics()[-1][2],
                           rand.statistics()[-1][2],
                           elitist.statistics()[-1][2]))
    print("DONE")
    with open(file, "w") as f:
        for s in statistics:
            print("{},{},{}".format(*s), file=f)


def run_john(population=100, generations=2000, runs=30, size=30,
             file="jb-stats.csv"):

    default = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=JB,
        mutation=UniformBitflipMutation(probability=0.10),
        crossover=NPointCrossover(probability=0.85, points=2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2))

    rand = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=JB,
        mutation=UniformBitflipMutation(probability=0.10),
        crossover=NPointCrossover(probability=0.85, points=2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=RandomImmigrants(immigrants=0.2))

    elitist = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=JB,
        mutation=UniformBitflipMutation(probability=0.10),
        crossover=NPointCrossover(probability=0.85, points=2),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=UniformBitflipMutation(
                                         probability=0.10)))
    run(default, rand, elitist, runs, file, size)


def run_rosenbrock(population=100, generations=2000, runs=30, dim=20,
                   file="rosenbrock-stats.csv"):

    sigma = [random.random() for _ in range(dim)]
    domain = [(-2.048, 2.048) for _ in range(dim)]

    default = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=UniformCrossover(probability=0.85, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2))

    rand = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=UniformCrossover(probability=0.85, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=RandomImmigrants(immigrants=0.2))

    elitist = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=UniformCrossover(probability=0.85, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=GaussianMutation(
                                         probability=0.05,
                                         sigma=sigma,
                                         domain=domain)))

    run(default, rand, elitist, runs, file, dim, domain, rosenbrock)


def run_quartic(population=100, generations=2000, runs=30, dim=20,
                file="quartic-stats.csv"):

    sigma = [random.random() for _ in range(dim)]
    domain = [(-1.28, 1.28) for _ in range(dim)]

    default = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.9, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2))

    rand = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.9, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=RandomImmigrants(immigrants=0.2))

    elitist = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.9, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=GaussianMutation(
                                         probability=0.05,
                                         sigma=sigma,
                                         domain=domain)))

    run(default, rand, elitist, runs, file, dim, domain, quartic)


def run_sphere(population=100, generations=2000, runs=30, dim=20,
               file="sphere-stats.csv"):

    sigma = [random.random() for _ in range(dim)]
    domain = [(-5.12, 5.12) for _ in range(dim)]

    default = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2))

    rand = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=RandomImmigrants(immigrants=0.2))

    elitist = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=GaussianMutation(
                                         probability=0.05,
                                         sigma=sigma,
                                         domain=domain)))

    run(default, rand, elitist, runs, file, dim, domain, sphere)


def run_step(population=100, generations=2000, runs=30, dim=20,
             file="step-stats.csv"):

    sigma = [random.random() for _ in range(dim)]
    domain = [(-5.12, 5.12) for _ in range(dim)]

    default = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2))

    rand = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=RandomImmigrants(immigrants=0.2))

    elitist = EvolutionaryAlgorithm(
        popsize=population,
        generations=generations,
        problem=Function,
        mutation=GaussianMutation(
            probability=0.05, sigma=sigma, domain=domain),
        crossover=ArithmeticCrossover(
            probability=0.85, alpha=0.7, domain=domain),
        selection=KTournamentSelection(size=population, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=ElitistImmigrants(immigrants=0.2,
                                     mutation=GaussianMutation(
                                         probability=0.05,
                                         sigma=sigma,
                                         domain=domain)))

    run(default, rand, elitist, runs, file, dim, domain, step)


if __name__ == "__main__":
    # run_john()
    run_rosenbrock()
    # run_quartic()
    # run_sphere()
    # run_step()
