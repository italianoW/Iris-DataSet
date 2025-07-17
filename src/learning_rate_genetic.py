import random
import treino
import util
import math
import numpy as np

POPULATION_SIZE = 20
chromossome_fitness_tuples = [0] * POPULATION_SIZE
dataset = util.download_dados()
atributos_teste,rotulos_teste,atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
already_used = []

def main():
    generation = [(random.randint(1, 100) / 100) for _ in range(POPULATION_SIZE)]
    for _ in range(20):
        chromossome_fitness_tuples = fitness_test(generation)
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

def fitness_test(current_generation):
    for i in range(POPULATION_SIZE):
        pesos, bias = treino.treinar(atributos_treino, rotulos_treino, 50, current_generation[i])
        epoch_accuracy, _ = util.avaliar(bias, pesos, atributos_teste, rotulos_teste)
        chromossome_fitness_tuples[i] = (current_generation[i], epoch_accuracy)
    chromossome_fitness_tuples.sort(key=lambda x:x[1])
    return chromossome_fitness_tuples

def probabilities_calculator(accuracy_array):
    inverted = [1 / a for a in accuracy_array]
    total = sum(inverted)
    probabilities = [x / total for x in inverted]
    return probabilities

def roulette_choice(chromossomes, probabilities):
    first = float(np.random.choice(chromossomes, p=probabilities))
    second = float(np.random.choice(chromossomes, p=probabilities))
    
    while first in already_used:
        first = float(np.random.choice(chromossomes, p=probabilities))
      
    while second in already_used or first == second:
        second = float(np.random.choice(chromossomes, p=probabilities))
          
    return first, second


def crossing_over(chromossomes, probabilities, new_chromossomes):
    already_used = []
    for _ in range((90 * POPULATION_SIZE) // 100):
        parent1, parent2 = roulette_choice(chromossomes, probabilities)
        already_used.append(parent1)
        already_used.append(parent2)
        
        child = (parent1 + parent2) / 2
        new_chromossomes.append(child)
    return new_chromossomes

def mutation(chromossomes):
    for _ in range((10 * POPULATION_SIZE) // 100):
        mutated = random.randint(0, POPULATION_SIZE - 1)
        if random.random() <= 0.5:
            chromossomes[mutated] = min((chromossomes[mutated] * 1.1), 1)
        else:
            chromossomes[mutated] = chromossomes[mutated] * 0.9
    
    return chromossomes

if __name__ == "__main__":
    main()