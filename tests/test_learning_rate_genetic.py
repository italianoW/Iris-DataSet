import pytest
import random
import numpy as np
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import learning_rate_genetic

def test_probabilities_calculator_sums_to_1():
    probs = learning_rate_genetic.probabilities_calculator([0.5, 0.5])
    assert pytest.approx(sum(probs), 0.01) == 1.0

def test_probabilities_calculator_empty():
    result = learning_rate_genetic.probabilities_calculator([])
    assert result == []

def test_probabilities_calculator_all_zeros():
    with pytest.raises(ZeroDivisionError):
        learning_rate_genetic.probabilities_calculator([0.0, 0.0])

def test_roulette_choice_valid_output():
    chromossomes = [1, 2, 3]
    probs = [0.2, 0.3, 0.5]
    p1, p2 = learning_rate_genetic.roulette_choice(chromossomes, probs)
    assert p1 in chromossomes and p2 in chromossomes
    assert p1 != p2

def test_roulette_choice_single_element():  # FALHANDO FALTA VERIFICAR SE SÓ EXISTE UM ELEMENTO
    with pytest.raises(ValueError):
        learning_rate_genetic.roulette_choice([1], [1.0])

def test_roulette_choice_empty():
    with pytest.raises(ValueError):
        learning_rate_genetic.roulette_choice([], [])

def test_crossing_over_returns_correct_size():
    chromossomes = list(range(20))
    probs = [1/20.0] * 20
    elite = chromossomes[:2]
    result = learning_rate_genetic.crossing_over(chromossomes, probs, elite[:])
    assert len(result) == 20

def test_mutation_changes_some_genes():
    chromossomes = [10] * 20
    mutated = learning_rate_genetic.mutation(chromossomes[:])
    assert chromossomes != mutated

def test_mutation_bounds_epochs():
    chromossomes = [250] * 20
    mutated = learning_rate_genetic.mutation(chromossomes[:])
    assert all(1 <= x <= 250 for x in mutated)

@patch('treino.treinar')
@patch('util.avaliar')
def test_envolve_mock(mock_avaliar, mock_treinar):
    mock_treinar.return_value = ([0.1, 0.2], 0.5)
    mock_avaliar.return_value = (0.9, None)

    best = learning_rate_genetic.envolve(5)
    assert isinstance(best, float)
    assert 0.01 <= best <= 0.3

def test_mutation_empty_input():
    result = learning_rate_genetic.mutation([])
    assert result == []

def test_mutation_single_element():
    result = learning_rate_genetic.mutation([100])
    assert isinstance(result, list)
    assert 1 <= result[0] <= 250