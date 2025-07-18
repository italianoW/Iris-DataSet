"""This file contains the training function for a neural network model"""
import math
import random

pesos = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(4)]
bias = [random.uniform(-0.1, 0.1) for _ in range(3)]

def treinar(atributos, rotulos, epocas, taxa_aprendizado):
    """Train a neural network model using the provided attributes and labels."""

    def one_hot(i, tamanho):
        """Convert a label index into a one-hot encoded vector."""
        vetor = [0] * tamanho
        vetor[i] = 1
        return vetor

    def relu(x):
        """Apply the ReLU activation function to a list of values."""
        return [max(0, v) for v in x]

    def softmax(x):
        """Calculate the softmax of a list of values."""
        max_x = max(x)
        e_x = [math.exp(i - max_x) for i in x]
        soma = sum(e_x)
        return [i / soma for i in e_x]

    rotulos_onehot = [one_hot(i, 3) for i in rotulos]


# === Treinamento ===

    for _ in range(epocas):

        for entrada, esperado in zip(atributos, rotulos_onehot):
            saida_linear = [bias[i] + sum(entrada[j] * pesos[j][i]
                                   for j in range(4))
                     for i in range(3)]
            saida = softmax(relu(saida_linear))


            for i in range(3):
                erro = saida[i] - esperado[i]
                for j in range(4):
                    pesos[j][i] -= taxa_aprendizado * erro * entrada[j]

                bias[i] -= taxa_aprendizado * erro

    return pesos, bias
