import random
from operator import itemgetter
import math
##Mutations
def muta_bin(indiv,prob_muta, muta_func):
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_func(cromo[i],prob_muta)
    return cromo


def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random.random()
    if value < prob_muta:
        g ^= 1
    return g


# discrete (include integers)

def muta_discrete(alphabet,indiv,prob_muta): #OK: requer arrays separando as letras (ex: ['A','B','C','D','E'])
    # Mutation by gene
    chromo = indiv[:]
    for i in range(len(indiv)):
        chromo[i] = muta_discrete_gene(alphabet, chromo[i],prob_muta)
    return chromo

def muta_discrete_gene(alphabet, gene, prob_muta):
    value = random.random()
    new_gene = gene
    if value < prob_muta:
        new_gene = random.choice(alphabet)
        while new_gene == gene:
            new_gene = random.choice(alphabet)
    return new_gene

# permutations

def muta_perm_swap(indiv, prob_muta): #OK
    """Swap mutation."""
    cromo = indiv[:]
    value = random.random()
    if value < prob_muta:
        index = random.sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        cromo[i1],cromo[i2] = cromo[i2], cromo[i1]
    return cromo

def muta_perm_insertion(indiv, prob_muta): #OK
    """ Insseertion mutation."""
    cromo = indiv[:]
    value = random.random()
    if value < prob_muta:
        index = random.sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index

        gene = cromo[i2]
        for i in range(i2,i1,-1):
            cromo[i] = cromo[i-1]
        cromo[i1+1] = gene
        
    return cromo

def muta_perm_scramble(indiv,prob_muta): #OK
    """ Scramble mutation."""
    cromo = indiv[:]
    value = random.random()
    if value < prob_muta:
        index = random.sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        scramble = cromo[i1:i2+1]
        random.shuffle(scramble) 
        cromo = cromo[:i1] + scramble + cromo[i2+1:]
    return cromo

def muta_perm_inversion(indiv,prob_muta): #OK
    """Invertion mutation."""
    cromo = indiv[:]
    value = random.random()
    if value < prob_muta:
        index = random.sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        inverte = []
        for elem in cromo[i1:i2+1]:
            inverte = [elem] + inverte
        cromo = cromo[:i1] + inverte + cromo[i2+1:]
    return cromo

# reals

def muta_reals(indiv, prob_muta, domain, sigma):
    cromo = indiv[:]
    for i in range(len(cromo)):
        cromo[i] = muta_reals_gene(cromo[i],prob_muta, domain[i], sigma[i])
    return cromo

# -- gaussian
def muta_reals_gene(gene,prob_muta, domain_gene, sigma_gene):
    value = random.random()
    new_gene = gene
    if value < prob_muta:
        muta_value = random.gauss(0,sigma_i)
        new_gene = gene + muta_value
        if new_gene < domain_gene[0]:
            new_gene = domain_gene[0]
        elif new_gene > domain_gene[1]:
            new_gene = domain_gene[1]
    return new_gene

# -- uniform	
def muta_reals_uni(indiv, prob_muta, domain):
    cromo = indiv[:]
    for i in range(len(cromo)):
        cromo[i] = muta_reals_uni_gene(cromo[i],prob_muta, domain[i])
    return cromo

def muta_reals_uni_gene(gene,prob_muta, domain_gene):
    value = random.random()
    new_gene = gene
    if value < prob_muta:
        new_gene = random.uniform(domain_gene[0],domain_gene[1])
    return new_gene



    ## Crossover

    # Operadores de Recombinação
# Genéricos
def one_point_cross(cromo_1, cromo_2,prob_cross): #OK
	value = random.random()
	if value < prob_cross:
		pos = random.randint(0,len(cromo_1))
		f1 = cromo_1[0:pos] + cromo_2[pos:]
		f2 = cromo_2[0:pos] + cromo_1[pos:]
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]
		

def two_points_cross(cromo_1, cromo_2,prob_cross): #OK
	value = random.random()
	if value < prob_cross:
		pc= random.sample(range(len(cromo_1)),2)
		pc.sort()
		pc1,pc2 = pc
		f1= cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
		f2= cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]
	
def uniform_cross(cromo_1, cromo_2,prob_cross): #OK
	value = random.random()
	if value < prob_cross:
		f1=[]
		f2=[]
		for i in range(0,len(cromo_1)):
			if random.random() < 0.5:
				f1.append(cromo_1[i])
				f2.append(cromo_2[i])
			else:
				f1.append(cromo_2[i])
				f2.append(cromo_1[i])

		return [f1,f2]
	else:
		return [cromo_1,cromo_2]

# Permutations
# OX - order crossover

def order_cross(cromo_1,cromo_2,prob_cross): #OK
	""" Order crossover."""
	size = len(cromo_1)
	value = random.random()
	if value < prob_cross:
		pc= random.sample(range(size),2)
		pc.sort()
		pc1,pc2 = pc
		f1 = [None] * size
		f2 = [None] * size
		f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
		f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
		for j in range(size):
			for i in range(size):
				if (cromo_2[j] not in f1) and (f1[i] == None):
					f1[i] = cromo_2[j]
					break
				

			for k in range(size):
				if (cromo_1[j] not in f2) and (f2[k] == None):
					f2[k] = cromo_1[j]
					break
				
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]
	
	
def pmx_cross(cromo_1,cromo_2,prob_cross): #OK 
	""" Partially mapped crossover."""
	size = len(cromo_1)
	value = random.random()
	if value < prob_cross:
		pc= random.sample(range(size),2)
		pc.sort()
		pc1,pc2 = pc
		f1 = [None] * size
		f2 = [None] * size
		f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
		f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
		# first offspring
		# middle part
		for j in range(pc1,pc2+1):
			if cromo_2[j] not in f1:
				pos_2 = j
				g_j_2 = cromo_2[pos_2]
				g_f1 = f1[pos_2]
				index_2 = cromo_2.index(g_f1)
				while f1[index_2] != None:
					index_2 = cromo_2.index(f1[index_2])
				f1[index_2] = g_j_2

		# remaining
		for k in range(size):
			if f1[k] == None:
				f1[k] = cromo_2[k]

		# secong offspring	
		# middle part
		for j in range(pc1,pc2+1):
			if cromo_1[j] not in f2:
				pos_1 = j
				g_j_1 = cromo_1[pos_1]
				g_f2 = f2[pos_1]
				index_1 = cromo_1.index(g_f2)
				while f2[index_1] != None:
					index_1 = cromo_1.index(f2[index_1])
				f2[index_1] = g_j_1

		# remaining
		for k in range(size):
			if f2[k] == None:
				f2[k] = cromo_1[k]
			
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]
	
	
# CYCLE
def cycle_cross(indiv_1,indiv_2,prob_cross): #OK
	""" Cycle crossover."""
	size = len(indiv_1)
	value = random.random()
	positions = [0]*size
	crosses = []
	if value < prob_cross:
		while(sum(positions)<size):
			#get first unoccupied place
			i = get_unoccupied(positions)
			temp1 = []
			while(True):
				positions[i] = 1
				temp1.append(i)
				i = indiv_1.index(indiv_2[i])
				if(i in temp1):
					crosses.append(temp1)
					break
		cycles_num = len(crosses)
		if(cycles_num <2):
			return indiv_1,indiv_2

		new_indiv_1 = get_decision(cycles_num)
		new_indiv_2 = get_decision(cycles_num,True)

		#mount  individuals
		individual1 = mount_individual(indiv_1,indiv_2,new_indiv_1,crosses,size)
		individual2 = mount_individual(indiv_1,indiv_2,new_indiv_2,crosses,size)
		
		return individual1 , individual2
	else:
		return indiv_1,indiv_2

def get_decision(cycles_num,inverse= False):
	decision = []
	if(inverse == False):
		for i in range(cycles_num):
			if(i%2):
				decision.append(1)
			else:
				decision.append(2)
	elif(inverse == True):
		for i in range(cycles_num):
			if(i%2):
				decision.append(2)
			else:
				decision.append(1)
	return decision

def mount_individual(indiv_1,indiv_2,structure,cycles,size):
	individual = [0]*size
	for ind,i in enumerate(cycles):
		for j in i:
			if (structure[ind] ==1 ):
				individual[j] = indiv_1[j]
			elif(structure[ind] ==2 ):
				individual[j] = indiv_2[j]
	return [individual,0]

def get_unoccupied(positions):
	for i in positions:
		if(i==0):
			return positions.index(i)
		
# reals (ainda não fiz código para incorporar estes)

def cross_reals_ari(alpha):
	def aritmetic(indiv_1,indiv_2,prob_cross):
		off_1 = indiv_1[0]
		off_2 = indiv_2[0]
		value = random.random()
		if value < prob_cross:
			size = len(indiv_1[0])
			off_1 = [ alpha * indiv_1[0][index] + (1 - alpha) * indiv_2[0][index] for index in range(size)]
			off_2 = [ alpha * indiv_2[0][index] + (1 - alpha) * indiv_1[0][index] for index in range(size)]
			return [[off_1,0],[off_2,0]]
		return [[off_1,0],[off_2,0]]
	return aritmetic
		

def heuristic(cromo_1,cromo_2,alpha,fit_func):
	"""Just one offspring!."""
	fit_cr_1 = fitness(fit_func,cromo_1)
	fit_cr_2 = fitness(fit_func,cromo_2)
	if fit_cr_1 > fit_cr_2:
		off = [ alpha* (cromo_1[index] - cromo_2[index]) + cromo_1[index] for index in range(len(cromo_1))]
	else:
		off = [ alpha* (cromo_2[index] - cromo_1[index]) + cromo_2[index] for index in range(len(cromo_1))]
	return off

## Tournament Selection
def tour_sel(t_size):
	def tournament(pop):
		size_pop= len(pop)
		mate_pool = []
		for i in range(size_pop):
			winner = tour(pop,t_size)
			mate_pool.append(winner)
		return mate_pool
	return tournament

def tour(population,size):
	"""Minimization Problem.Deterministic"""
	pool = random.sample(population, size)
	pool.sort(key=itemgetter(1))
	return pool[0]

## Russian Roulette Selection
def roulette_wheel(population, numb):
    """ Select numb individuals from the population
    according with their relative fitness. MAX. """
    pop = population[:]
    pop.sort(key=itemgetter(1))
    total_fitness = sum([indiv[1] for indiv in pop])
    mate_pool = []
    for i in range(numb):
        value = random.uniform(0,1)
        index = 0
        total = pop[index][1]/ float(total_fitness)
        while total < value:
            index += 1
            total += pop[index][1]/ float(total_fitness)
        mate_pool.append(pop[index])
    return mate_pool

##JB
def jb_fitness(indiv, alpha, beta):
	""" João Brandão's Numbers Problem - Fitness Function

	alpha = Weight of the reward associated with the number of
            correct choices of numbers for the final set
	beta = Weight of the penalty associated with the number of
			incorrect choices of numbers (that violate the problem
			constraints) for the final set
"""
	violations = 0
	for i in range(1, len(indiv)):
		v, lim = 0, min(i, len(indiv) - i - 1)
		for j in range(1, lim + 1):
			if (indiv[i] == 1) and (indiv[i - j] == 1) \
					and (indiv[i + j] == 1):
				v += 1
		violations += v
	return alpha * sum(indiv) - beta * violations,	violations




def cromo_bin(size):
	indiv = [random.randint(0,1) for i in range(size)]
	return indiv

def gen_pop(benchmark_problem,n_indiv,length_indivs,alpha,beta):
	pop=[]
	if benchmark_problem == "JBnumbers":
		for i in range(n_indiv):
			
			new_indiv = cromo_bin(length_indivs)
			fitness_value, violations = jb_fitness(new_indiv, alpha, beta)
			"""versão com população de individuos sem violações (tamanho 100 fica suuuuper lento)
			while(violations != 0):
				new_indiv = cromo_bin(length_indivs,domain)
				fitness_value, violations = jb_fitness(new_indiv, alpha, beta)
				"""
			pop.append(new_indiv)
		
		return pop, jb_fitness
    #elif benchmark_problem == "functions":
        #?


def select_best_fitness(population,n_selected):
	l = len(population)
	for i in range(l-1):
		if (population[i][1] < population[i + 1][1]):
			elem = population[i]
			population[i]= population[i + 1]
			population[i + 1]= elem
	return population[0:n_selected]


def test_parts(benchmark_problem,n_indiv,length_indivs,alpha,beta):
	pop,fitness_func = gen_pop(benchmark_problem,n_indiv,length_indivs,alpha,beta)
	data=[]
	for i in pop:
		[fitness,vio] = fitness_func(i,alpha,beta)
		data.append([i,fitness])
	
	return data

def select_parents(population,selec_func,n_selected):
	if selec_func == "tournament":
		return tour_sel(n_selected)(population)
	elif selec_func == "roulette":
		return roulette_wheel(population,n_selected)
	elif selec_func == "best_fitness":
		return select_best_fitness(population,n_selected)

def generate_offspring(parents,cross_function,cross_rate,n_generations,fitness,alpha,beta):
	offspring_fam = []
	offspring_gen = []
	offspring_unique = []
	offspring_ = []
	print("parents:",parents)
	for i in range(n_generations):
		offspring_gen = []
		for j in range(math.ceil(len(parents)/2)):
			print("n matches=",math.ceil(len(parents)/2), "len(parents)=",len(parents))
			match = random.sample(parents,2)
			parents_=select_best_fitness(match,2)
			parent1, parent2, fitness_1, fitness_2 = parents_[0][0], parents_[1][0], parents_[0][1], parents_[1][1]
			child1, child2=cross_function(parent1,parent2,cross_rate)
			fitness_child1,viol = fitness(child1,alpha,beta)
			fitness_child2,viol = fitness(child2,alpha,beta)
			#print(fitness_child1,fitness_child2)
			#print("fitness child1: ",fitness_child1,"\nfitness child2: ",fitness_child2,"\nfitness parent1: ",fitness_1,"\nfitness parent2: ",fitness_2)
			if fitness_child1 > fitness_2 and child1 not in offspring_gen:
				offspring_gen.append(child1)
			if fitness_child2 > fitness_2 and child2 not in offspring_gen:
				offspring_gen.append(child2)
			print("offspring_gen: ",offspring_gen)
		if offspring_gen != []:	
			offspring_.append(offspring_gen) 
		print("\noffspring: ",offspring_)
	"""
	for i in range(len(offspring_gen)):
		offspring_unique = []
		for j in range(len(offspring_gen[i])):
			if offspring_gen[i][j] not in offspring_unique:
				offspring_unique.append(offspring_gen[i][j])
				print("\noffspring unique=",offspring_unique)
		if offspring_unique:
			offspring.append([offspring_unique])
	"""
	#print("\noffspring=",offspring)
	return offspring_


def obtain_data(n_runs,benchmark_problem,n_indiv,length_indivs,selec_func,n_selec_parents,n_generations,n_imigrants,n_final_indiv,mut_type,cross_type,alpha:float=1.0,beta:float=1.5,cross_rate=0.9,mut_rate=0.1):
	pop=[]
	parents=[]
	offspring=[]
	data=[]
	for i in range(n_runs):
		#generate population
		pop,fitness_func = gen_pop(benchmark_problem,n_indiv,length_indivs,alpha,beta)

		data_=[]
		offspring_=[]
		parents_=[]

		#add fitness to population
		for i in pop:
			fitness_par,vio = fitness_func(i,alpha,beta)
			data_.append([i,fitness_par])
		
		#select parents	(cada pai tem 2 parametros: individuo e fitness)
		parents_=select_parents(data_,selec_func,n_selec_parents)

		#crossover: dá com todas menos os reals
		offspring_=generate_offspring(parents_,cross_type,cross_rate,n_generations,fitness_func,alpha,beta)
		"""for f in range(len(offspring_)):
			for i in range(len(offspring_[f])):
				fitness_off,vio = fitness_func(offspring_[f][i],alpha,beta)
				offspring_.append([offspring_[f],fitness_off])
		"""
		#mutation (undone)



		data.append(data_)
		parents.append(parents_)
		offspring.append(offspring_) #resulta numa matriz dividida por runs e gerações. 

	return data, parents, offspring
		



if __name__ == '__main__':
	pop,parents,offspring = obtain_data(2,"JBnumbers",10,10,"best_fitness",5,7,5,5,"jioh",one_point_cross,alpha=1.0,beta=1.5,cross_rate=0.9,mut_rate=0.1)
	print("\noffspring=",offspring)
