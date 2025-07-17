import random
import treino
import util
import math
import numpy as np

random.seed(42)
np.random.seed(42)
POPULATION_SIZE = 20
chromossome_fitness_tuples = [0] * POPULATION_SIZE
dataset = util.download_dados()
atributos_teste,rotulos_teste,atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
already_used = []

def envolve(learning_rate):
    generation = [random.randint(1, 200) for _  in range(POPULATION_SIZE)]
    
    for _ in range(20):
        chromossome_fitness_tuples = fitness_test(generation, learning_rate)
        sorted_generation = [x[0] for x in chromossome_fitness_tuples]
        sorted_fit_scores = [x[1] for x in chromossome_fitness_tuples]
        

        new_generation = sorted_generation[:(10 * POPULATION_SIZE) // 100]

        probabilities = probabilities_calculator(sorted_fit_scores)

        new_generation.extend(crossing_over(sorted_generation, probabilities, new_generation))

        new_generation = mutation(new_generation)
        
        generation = new_generation
        
        
    return generation[0]

def fitness_test(current_generation, train_learn_rt):
    for i in range(POPULATION_SIZE):
        accs = []
        for _ in range(2):  # repeat 3 times per individual
            pesos, bias = treino.treinar(atributos_treino, rotulos_treino, current_generation[i], train_learn_rt)
            acc, _ = util.avaliar(bias, pesos, atributos_teste, rotulos_teste)
            accs.append(acc)
        avg_accuracy = sum(accs) / len(accs)
        chromossome_fitness_tuples[i] = (current_generation[i], avg_accuracy)
    chromossome_fitness_tuples.sort(key=lambda x: x[1], reverse=True)
    return chromossome_fitness_tuples

def probabilities_calculator(accuracy_array):
    exp_scores = np.exp(accuracy_array)
    probabilities = exp_scores / exp_scores.sum()
    return probabilities.tolist()

def roulette_choice(chromossomes, probabilities):
    attempts = 0
    first = int(np.random.choice(chromossomes, p=probabilities))
    second = int(np.random.choice(chromossomes, p=probabilities))
         
    while first in already_used:
        attempts += 1
        first = int(np.random.choice(chromossomes, p=probabilities))
        if attempts > 2: break
            
    while second in already_used or first == second:
        attempts += 1
        second = int(np.random.choice(chromossomes, p=probabilities))
        if attempts > 2: break
          
    return first, second


def crossing_over(chromossomes, probabilities, new_chromossomes):
    global already_used
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
        chromossomes[mutated] = min(250, math.ceil(chromossomes[mutated] * 1.1 + random.uniform(-3, 3)))
    
    return chromossomes