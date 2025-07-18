"""Implements a genetic algorithm to optimize the number of epochs."""

import math
import random
import numpy as np
import treino
import util

random.seed(42)
np.random.seed(42)
POPULATION_SIZE = 20
chromossome_fitness_tuples = [(0, 0)] * POPULATION_SIZE
dataset = util.download_dados()
atributos_teste,rotulos_teste,attr_trein,rot_treino = util.divisao_treino_teste(dataset)
already_used = []

def envolve(learning_rate):
    """Run the genetic algorithm to find the best number of epochs."""
    generation = [random.randint(1, 200) for _  in range(POPULATION_SIZE)]

    for _ in range(20):
        fitness_test(generation, learning_rate)
        sorted_generation = [chrmss for chrmss, _ in chromossome_fitness_tuples]
        sorted_fit_scores = [fitness for _, fitness in chromossome_fitness_tuples]


        new_generation = sorted_generation[:(10 * POPULATION_SIZE) // 100]

        probabilities = probabilities_calculator(sorted_fit_scores)

        new_generation.extend(crossing_over(sorted_generation, probabilities, new_generation))

        new_generation = mutation(new_generation)

        generation = new_generation

    return generation[0]

def fitness_test(curr_gen, train_learn_rt):
    """Evaluate the fitness of each chromosome in the current generation."""
    for i in range(POPULATION_SIZE):
        accs = []
        for _ in range(2):
            pesos, bias = treino.treinar(attr_trein, rot_treino, curr_gen[i], train_learn_rt)
            acc, _ = util.avaliar(bias, pesos, atributos_teste, rotulos_teste)
            accs.append(acc)
        avg_accuracy = sum(accs) / len(accs)
        chromossome_fitness_tuples[i] = (curr_gen[i], avg_accuracy)
    chromossome_fitness_tuples.sort(key=lambda x: x[1], reverse=True)

def probabilities_calculator(accuracy_array):
    """Calculate the probabilities for each chromosome based on their fitness scores."""
    exp_scores = np.exp(accuracy_array)
    probabilities = exp_scores / exp_scores.sum()
    return probabilities.tolist()

def roulette_choice(chromossomes, probabilities):
    """Select two parents using roulette wheel selection."""
    attempts = 0
    first = int(np.random.choice(chromossomes, p=probabilities))
    second = int(np.random.choice(chromossomes, p=probabilities))

    while first in already_used:
        attempts += 1
        first = int(np.random.choice(chromossomes, p=probabilities))
        if attempts > 2:
            break

    while second in already_used or first == second:
        attempts += 1
        second = int(np.random.choice(chromossomes, p=probabilities))
        if attempts > 2:
            break

    return first, second


def crossing_over(chromossomes, probabilities, new_chromossomes):
    """Perform crossover to create new chromosomes from selected parents."""
    already_used.clear()
    for _ in range((90 * POPULATION_SIZE) // 100):
        parent1, parent2 = roulette_choice(chromossomes, probabilities)
        already_used.append(parent1)
        already_used.append(parent2)

        child = (parent1 + parent2) // 2
        new_chromossomes.append(child)
    return new_chromossomes

def mutation(chromss):
    """Mutate the chromosomes by slightly adjusting their values."""
    for _ in range((10 * POPULATION_SIZE) // 100):
        mutated = random.randint(0, POPULATION_SIZE - 1)
        chromss[mutated] = min(250, math.ceil(chromss[mutated] * 1.1 + random.uniform(-3, 3)))

    return chromss
