#!/usr/bin/env python3
import random
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

# João Brandão's Numbers Benchmark Probelm
from eaim.benchmark import JB

# Function Optimization Benchmark Problem
from eaim.benchmark import Function
from eaim.benchmark import sphere, rosenbrock, step, quartic
from eaim.benchmark import rastringin, schewefel, griewank


def test_JB(size, popsize, generations, problem, mutation_type, mutation_prob, crossover_type, crossover_prob, selection_type, n_points, immigrant_insertion=None):

    dim = [2, 3, 4]

    ea = EvolutionaryAlgorithm(
        popsize=popsize,
        generations=generations,
        problem=problem,
        mutation=mutation_type(probability=mutation_prob),
        crossover=crossover_type(probability=crossover_prob, points=n_points),
        selection=selection_type(size=100, k=3),
        survivors=Elitism(elite=0.2),
        immigrants=immigrant_insertion)

    ea(size)
    stats = ea.statistics()
    savedata_JB(stats, JB, size, popsize, generations, mutation_type, mutation_prob,
                crossover_type, crossover_prob, n_points, selection_type, immigrant_insertion)


def savedata_JB(stats, benchmark, size, popsize, generations, mutation_type,
                mutation_prob, crossover_type, crossover_prob, n_point, selection_type, immigrant_insertion):
    if selection_type == KTournamentSelection:
        selec_type = "KTournamentSelection"
    else:
        selec_type = "RouletteWheelSelection"

    with open('../data/%s_size%d_popsize%d_gen%d__muttype_%s_mutprob%.2f_crosstype_%s_crossprob%.2f_npoints%d_seltype_%s_imminser_%s.txt'
              % (benchmark.__name__, size, popsize, generations, mutation_type.__name__, mutation_prob, crossover_type.__name__, crossover_prob, n_point, selec_type, immigrant_insertion), 'a') as fp:
        filepath = os.path.abspath(fp.name)
        if os.stat(filepath).st_size == 0:
            fp.write(
                "Seed,Individual,Fitness,Mean,Standard_Deviation,Variance,Median,Mode\n")
        fp.write(','.join('%s' % x for x in stats[-1]))
        fp.write('\n')
        fp.close()

# Da resultados estranhos nas estatisticas


def test_functions(popsize, generations, problem, mutation_type, mutation_prob, crossover_type, crossover_prob, selection_type, n_points, immigrant_insertion=None):

    dim = 20
    s = [random.random() for _ in range(dim)]
    functions = [[sphere, [(-5.12, 5.12) for _ in range(dim)]],
                 [rosenbrock, [(-2.048, 2.048) for _ in range(dim)]],
                 [step, [(-5.12, 5.12) for _ in range(dim)]],
                 [quartic, [(-1.28, 1.28) for _ in range(dim)]],
                 [rastringin, [(-5.12, 5.12) for _ in range(dim)]],
                 [schewefel, [(-500, 500) for _ in range(dim)]],
                 [griewank, [(-600, 600) for _ in range(dim)]]]

    for i in range(len(functions)):
        ea = EvolutionaryAlgorithm(
            popsize=popsize,
            generations=generations,
            problem=Function,
            mutation=mutation_type(
                probability=mutation_prob, sigma=s, domain=functions[i][1]),
            crossover=crossover_type(
                probability=crossover_prob, points=n_points),
            selection=selection_type(size=100, k=3),
            survivors=Elitism(elite=0.2),
            immigrants=immigrant_insertion)

        ea(dim, functions[i][1], (functions[i][0]))
        stats = ea.statistics()
        savedata(stats, functions[i][0], 20, popsize, generations, mutation_type, mutation_prob,
                 crossover_type, crossover_prob, n_points, selection_type, immigrant_insertion)


def savedata(stats, benchmark, dim, popsize, generations, mutation_type, mutation_prob, crossover_type, crossover_prob, n_point, selection_type, immigrant_insertion):
    if selection_type == KTournamentSelection:
        selec_type = "KTournamentSelection"
    else:
        selec_type = "RouletteWheelSelection"

    with open('../data/%s_dim%d_popsize%d_gen%d__muttype_%s_mutprob%.2f_crosstype_%s_crossprob%.2f_npoints%d_seltype_%s_imminser_%s.txt'
              % (benchmark.__name__, dim, popsize, generations, mutation_type.__name__, mutation_prob, crossover_type.__name__, crossover_prob, n_point, selec_type, immigrant_insertion), 'a') as fp:
        filepath = os.path.abspath(fp.name)
        if os.stat(filepath).st_size == 0:
            fp.write(
                "Seed,Individual,Fitness,Mean,Standard_Deviation,Variance,Median,Mode\n")
        fp.write(','.join('%s' % x for x in stats[-1]))
        fp.write('\n')
        fp.close()

    # TODO: function that takes statistics and plots them over generation
    # fitnes of the best and average over all the generations

    # Function that runs the problem with multiple seeds and saves the
    # infomation of multiple runs in a file (in this case
    # john(size, seed=XXXXXX)). Its just a loop basically


if __name__ == "__main__":
    generations = [1000, 1500, 2000]
    population_dim = [50, 100]

    crossover_type = [NPointCrossover, UniformCrossover]
    crossover_prob = [0.85, 0.9, 0.95]
    mutation_prob = [0.05, 0.1, 0.15]

    selection_type = [KTournamentSelection, RouletteWheelSelection]

    for dim_pop in population_dim:
        for gen in generations:
            for mp in mutation_prob:
                for cp in crossover_prob:
                    for ct in crossover_type:
                        for st in selection_type:
                            for _ in range(1):
                                test_JB(size=30,
                                        popsize=dim_pop,
                                        generations=gen,
                                        problem=JB,
                                        mutation_type=UniformBitflipMutation,
                                        mutation_prob=mp,
                                        crossover_type=ct,
                                        crossover_prob=cp,
                                        selection_type=st,
                                        immigrant_insertion=None,
                                        n_points=2)
