import pytest
import random
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import epochs_genetic

def test_fitness_test():
    generation = [random.randint(1, 250) for _ in range(epochs_genetic.POPULATION_SIZE)]
    epochs_genetic.fitness_test(generation, 0.01)
    
    result = epochs_genetic.chromossome_fitness_tuples
    assert len(result) == epochs_genetic.POPULATION_SIZE

    for cromossomo, fitness in result:
        assert isinstance(cromossomo, int)
        assert isinstance(fitness, float)
        assert 0 <= fitness <= 1

    fitness_vals = [f for _, f in result]
    assert fitness_vals == sorted(fitness_vals, reverse=True)

@patch('epochs_genetic.util.avaliar')
@patch('epochs_genetic.treino.treinar')
@patch('epochs_genetic.util.divisao_treino_teste')
@patch('epochs_genetic.util.download_dados')
def test_envolve_mock_epochs(mock_download, mock_divisao, mock_treinar, mock_avaliar):

    mock_download.return_value = 'mock_dataset'
    mock_divisao.return_value = (
        [[0.1, 0.2]],  # atributos_teste
        [1],           # rotulos_teste
        [[0.3, 0.4]],  # attr_trein
        [0]            # rot_treino
    )

    mock_treinar.return_value = (
        [0.1, 0.2],  # pesos_entrada_oculta
        [0.3, 0.4],  # saida_oculta_pesos
        0.1,         # bias_oculta
        0.2          # bias_saida
    )

    mock_avaliar.return_value = (0.85, None, None)

    best = epochs_genetic.envolve(learning_rate=0.01)

    assert isinstance(best, int)
    assert 1 <= best <= 250

def test_probabilities_calculator():
    accuracies = [0.5, 0.25, 0.25]
    probs = epochs_genetic.probabilities_calculator(accuracies)
    assert len(probs) == len(accuracies)
    assert abs(sum(probs) - 1) < 1e-8
    assert probs[0] > probs[1]

def test_probabilities_calculator_empty():
    result = epochs_genetic.probabilities_calculator([])
    assert result == []

def test_roulette_choice():
    cromossomos = [5, 10, 15, 20]
    probs = [0.25] * 4
    c1, c2 = epochs_genetic.roulette_choice(cromossomos, probs)
    assert c1 in cromossomos
    assert c2 in cromossomos

def test_roulette_choice_empty():
    with pytest.raises(Exception):
        epochs_genetic.roulette_choice([], [])

def test_crossing_over_geracao_cheia():
    cromossomos = list(range(20, 40))
    probs = [1 / 20] * 20
    elite = cromossomos[:2]
    nova_geracao = epochs_genetic.crossing_over(cromossomos, probs, elite[:])
    assert len(nova_geracao) == epochs_genetic.POPULATION_SIZE
    for epocas in nova_geracao:
        assert isinstance(epocas, int)
        assert 1 <= epocas <= 250

def test_mutation_altera_valores():
    cromossomos = [10] * epochs_genetic.POPULATION_SIZE
    mutados = epochs_genetic.mutation(cromossomos[:])
    assert any(a != b for a, b in zip(cromossomos, mutados))

def test_mutation_limites():
    cromossomos = [250] * epochs_genetic.POPULATION_SIZE
    mutados = epochs_genetic.mutation(cromossomos[:])
    assert all(1 <= v <= 250 for v in mutados)