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


def test_JB(size,popsize,generations,problem,mutation_type,mutation_prob,crossover_type,crossover_prob,selection_type,n_points,immigrant_insertion=None):
     
    
    dim=[2,3,4]
        
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
    savedata(stats,JB, popsize, generations,mutation_type, mutation_prob,crossover_type, crossover_prob, n_points, selection_type, immigrant_insertion)

## Da resultados estranhos nas estatisticas
def test_functions(popsize,generations,problem,mutation_type,mutation_prob,crossover_type,crossover_prob,selection_type,n_point,immigrant_insertion=None):
     
    dim=[2,3,4]
    for l in range(len(dim)):
        s = [random.random() for _ in range(l)]
        functions = [[sphere,[(-5.12, 5.12) for _ in range(l)]], 
                    [rosenbrock,[(-2.048, 2.048) for _ in range(l)]], 
                    [step,[(-5.12, 5.12) for _ in range(l)]], 
                    [quartic,[(-1.28, 1.28) for _ in range(l)]], 
                    [rastringin,[(-5.12, 5.12) for _ in range(l)]], 
                    [schewefel,[(-500, 500) for _ in range(l)]], 
                    [griewank,[(-600, 600) for _ in range(l)]]]
    for i in range(len(functions)):
        ea = EvolutionaryAlgorithm(
            popsize=popsize,
            generations=generations,
            problem=problem,
            mutation=mutation_type(probability=mutation_prob,sigma=s,domain=functions[i][1]),
            crossover=crossover_type(probability=crossover_prob, points=n_point),
            selection=selection_type,
            survivors=Elitism(elite=0.2),
            immigrants=immigrant_insertion)
        for d in dim:
            ea(d,functions[i][1], (functions[i][0]))
            stats = ea.statistics()
            print(stats[-1])
            savedata(stats,functions[i][0], popsize, generations,mutation_type, mutation_prob, crossover_type, crossover_prob, n_point, selection_type,immigrant_insertion)


def savedata(stats,benchmark,dim, generations,mutation_type, mutation_prob, crossover_type,crossover_prob, n_point, selection_type,immigrant_insertion):
    if selection_type == KTournamentSelection:
        selec_type = "KTournamentSelection"
    else:
        selec_type = "RouletteWheelSelection"
    
    
    with open('../data/%s_dim%d_gen%d__muttype_%s_mutprob%.2f_crosstype_%s_crossprob%.2f_npoints%d_seltype_%s_imminser_%s.txt' 
            % (benchmark.__name__,dim, generations,mutation_type.__name__, mutation_prob, crossover_type.__name__,crossover_prob, n_point, selec_type,immigrant_insertion), 'a') as fp:
            filepath = os.path.abspath(fp.name)
            if os.stat(filepath).st_size == 0:
                    fp.write("Seed,Individual,Fitness,Mean,Standard_Deviation,Variance,Median,Mode\n")
            fp.write(','.join('%s' % x for x in stats[-1]))
            fp.write('\n')
            fp.close()

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


if __name__ == "__main__":
    
    #benchmark_problem=[JB,Function]
    generations=[500,1000,1500] #acabei por por assim pa ser mais rapido
    sizeindiv=[50,100,150]

    crossover_type=[NPointCrossover, ArithmeticCrossover, UniformCrossover]
    crossover_prob=[0.85,0.9,0.95]
    mutation_type=[UniformBitflipMutation,GaussianMutation,UniformMutation]
    mutation_prob=[0.05,0.1,0.15]

    n_points=[1,2]
    population_dim=[100,200,300]
    selection_type=[KTournamentSelection, RouletteWheelSelection]
    
    immigrant_insertion=[None,ElitistImmigrants(immigrants=0.2,mutation=UniformBitflipMutation(probability=0.2)),RandomImmigrants(immigrants=0.2)]
    for si in sizeindiv:
        for dim_pop in population_dim:
            for gen in generations:
                for mp in mutation_prob:
                    for cp in crossover_prob:
                        for st in selection_type:
                            for ii in immigrant_insertion:
                                if si== 50 and dim_pop == 100 and gen == 500 and mp==0.05 and cp== 0.85 and st == KTournamentSelection and ii == None:
                                    #estava a dar tentar dar skip do 1o ciclo pq já tenho os dados.
                                    break
                                for _ in range(30):
                                    test_JB(size=si,
                                            popsize=dim_pop,
                                            generations=gen,
                                            problem=JB,
                                            mutation_type=UniformBitflipMutation,
                                            mutation_prob=mp,
                                            crossover_type=NPointCrossover,
                                            crossover_prob=cp,
                                            selection_type=st,
                                            immigrant_insertion=ii,
                                            n_points= 1)
                                    test_JB(size=si,
                                            popsize=dim_pop,
                                            generations=gen,
                                            problem=JB,
                                            mutation_type=UniformBitflipMutation,
                                            mutation_prob=mp,
                                            crossover_type=NPointCrossover,
                                            crossover_prob=cp,
                                            selection_type=st,
                                            immigrant_insertion=ii,
                                            n_points= 2)
                                    