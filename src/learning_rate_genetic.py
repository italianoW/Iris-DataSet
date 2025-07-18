"""Implements a genetic algorithm to optimize the learning rate."""

import random
import numpy as np
import treino
import util

random.seed(42)
np.random.seed(42)
POPULATION_SIZE = 20
chromossome_fitness_tuples = [0] * POPULATION_SIZE
dataset = util.download_dados()
atributos_teste,rotulos_teste,attr_train,rtl_train = util.divisao_treino_teste(dataset)
already_used = []

def envolve(epochs):
    """Run the genetic algorithm to find the best learning rate."""
    generation = [round(random.uniform(0.01, 0.3), 5) for _ in range(POPULATION_SIZE)]
    for _ in range(20):
        fitness_test(generation, epochs)
        sorted_generation = [chrmss for chrmss, _ in chromossome_fitness_tuples]
        sorted_fit_scores = [fitness for _, fitness in chromossome_fitness_tuples]

        new_generation = sorted_generation[:(10 * POPULATION_SIZE) // 100]
        probabilities = probabilities_calculator(sorted_fit_scores)
        new_generation.extend(crossing_over(sorted_generation, probabilities, new_generation))

        new_generation = mutation(new_generation)

        generation = new_generation

    return generation[0]

def fitness_test(crrt_gen, train_epochs):
    """Evaluate the fitness of each chromosome in the current generation."""
    for i in range(POPULATION_SIZE):
        accs = []
        for _ in range(2):
            pesos_entrada_oculta, saida_oculta_pesos, bias_oculta, bias_saida = treino.treinar(attr_train, rtl_train, train_epochs, crrt_gen[i])
            acc, _, _ = util.avaliar(pesos_entrada_oculta, bias_oculta, saida_oculta_pesos, bias_saida, atributos_teste, rotulos_teste)
            accs.append(acc)
        avg_accuracy = sum(accs) / len(accs)
        chromossome_fitness_tuples[i] = (crrt_gen[i], avg_accuracy)
    chromossome_fitness_tuples.sort(key=lambda x:x[1], reverse=True)

def probabilities_calculator(accuracy_array):
    """Calculate the probabilities for each chromosome based on their fitness scores."""
    total = sum(accuracy_array)
    probabilities = [x / total for x in accuracy_array]
    return probabilities

def roulette_choice(chromossomes, probabilities):
    """Select two parents using roulette wheel selection."""
    if len(chromossomes) < 2 or len(probabilities) < 2:
        raise ValueError("At least two chromosomes and probabilities are required for selection.")

    attempts = 0
    first = round(float(np.random.choice(chromossomes, p=probabilities)), 5)
    second = round(float(np.random.choice(chromossomes, p=probabilities)), 5)

    while first in already_used:
        attempts += 1
        first = round(float(np.random.choice(chromossomes, p=probabilities)), 5)
        if attempts > 2:
            break

    while second in already_used or first == second:
        attempts += 1
        second = round(float(np.random.choice(chromossomes, p=probabilities)), 5)
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

        child = round(float((parent1 + parent2) / 2), 5)
        new_chromossomes.append(child)
    return new_chromossomes

def mutation(chromossomes):
    """Mutate the chromosomes by slightly adjusting their values."""
    if chromossomes == []:
        return []

    for _ in range((10 * POPULATION_SIZE) // 100):
        mutated = random.randint(0, len(chromossomes) - 1)
        if random.random() <= 0.5:
            chromossomes[mutated] = round(min((chromossomes[mutated] * 1.1), 1), 5)
        else:
            chromossomes[mutated] = round((chromossomes[mutated] * 0.9), 5)

    return chromossomes
