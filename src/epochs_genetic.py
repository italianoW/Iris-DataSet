import random
import treino
import util
import math
import numpy as np

POPULATION_SIZE = 20
generation = [random.randint(1, 250) for _ in range(POPULATION_SIZE)]
chromossome_fitness_tuples = [0] * POPULATION_SIZE
dataset = util.download_dados()
atributos_teste,rotulos_teste,atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
already_used = []

def main():
    for _ in range(20):
        chromossome_fitness_tuples = fitness_test()
        sorted_generation = [x[0] for x in chromossome_fitness_tuples]
        sorted_fit_scores = [x[1] for x in chromossome_fitness_tuples]

        new_generation = sorted_generation[:(10 * POPULATION_SIZE) // 100]

        probabilities = probabilities_calculator(sorted_fit_scores)

        new_generation.extend(crossing_over(sorted_generation, probabilities, new_generation))

        new_generation = mutation(new_generation)
        
        generation = new_generation
        print(generation)
        
    print(generation[0])
    return generation[0]

def fitness_test():
    for i in range(POPULATION_SIZE):
        _, erro_medio_final = treino.treinar(atributos_treino, rotulos_treino, generation[i])
        chromossome_fitness_tuples[i] = (generation[i], erro_medio_final)
    chromossome_fitness_tuples.sort(key=lambda x:x[1])
    return chromossome_fitness_tuples

def probabilities_calculator(generation):
    inverted = [(1 / v) for v in generation]
    total = sum(inverted)
    probabilities = [x / total for x in inverted]
    return probabilities

def roulette_choice(chromossomes, probabilities):
    
    first = int(np.random.choice(chromossomes, p=probabilities))
    second = int(np.random.choice(chromossomes, p=probabilities))
    
    while first in already_used:
        first = int(np.random.choice(chromossomes, p=probabilities))
      
    while second in already_used or first == second:
        second = int(np.random.choice(chromossomes, p=probabilities))
          
    return first, second


def crossing_over(chromossomes, probabilities, new_chromossomes):
    already_used = []
    for _ in range((90 * POPULATION_SIZE) // 100):
        parent1, parent2 = roulette_choice(chromossomes, probabilities)
        already_used.append(parent1)
        already_used.append(parent2)
        
        child = (parent1 + parent2) // 2
        new_chromossomes.append(child)
    return new_chromossomes

def mutation(chromossomes):
    for _ in range((10 * POPULATION_SIZE) // 100):
        mutated = random.randint(0, POPULATION_SIZE - 1)
        chromossomes[mutated] = math.ceil(chromossomes[mutated] * 1.1)
    
    return chromossomes


print(generation)
_, erro_medio_final = treino.treinar(atributos_treino, rotulos_treino, generation[0])
print(erro_medio_final)
print('changes')

if __name__ == "__main__":
    main()