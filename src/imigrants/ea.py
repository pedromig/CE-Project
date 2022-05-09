

def ea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross,
        sel_parents, recombination, mutation, sel_survivors, fitness_func):
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    indivs, avg = [], []

    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    indivs.append(best_pop(populacao)[1])
    avg.append(avg_pop(populacao))

    for i in range(numb_generations):
        # sparents selection
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, size_pop-1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1, indiv_2, prob_cross)
            progenitores.extend(filhos)
        # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        indivs.append(best_pop(populacao)[1])
        avg.append(avg_pop(populacao))
    return best_pop(populacao), indivs, avg
