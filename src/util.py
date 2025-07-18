"""Utility functions for processing the Iris dataset."""

import math
import os
import random
import pandas as pd

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "iris.data.csv")
def download_dados():
    """Download the Iris dataset from a CSV file (originated by the origial base data)."""
    return pd.read_csv(CSV)

def divisao_treino_teste(dados):
    """Divide the dataset into training and testing sets."""

    random.seed(42)
    lista = dados.values.tolist()
    treino,teste = [], []
    attr_test, rotulos_teste, atributos_treino, rotulos_treino = [], [], [], []
    lista.insert(0,lista[0])

    for i in range(3):
        for j in range(40):
            treino.append(lista[i*50 + j])

    for i in range(3):
        for j in range(10):
            teste.append(lista[i*50 + 33 + j])

    c1 = 33
    c2 = 33
    c3 = 33
    for item in lista:
        if item[4] == "Iris-setosa":
            if c1 > 0:
                c1 -= 1
                rotulos_treino.append(0)
                atributos_treino.append(item[:4])
            else:
                rotulos_teste.append(0)
                attr_test.append(item[:4])

        if item[4] == "Iris-versicolor":
            if c2 > 0:
                c2 -= 1
                rotulos_treino.append(1)
                atributos_treino.append(item[:4])
            else:
                rotulos_teste.append(1)
                attr_test.append(item[:4])

        if item[4] == "Iris-virginica":
            if c3 > 0:
                c3 -= 1
                rotulos_treino.append(2)
                atributos_treino.append(item[:4])
            else:
                rotulos_teste.append(2)
                attr_test.append(item[:4])

    return attr_test,rotulos_teste,atributos_treino,rotulos_treino

def avaliar(pesos_in_hide, bias_oculta, out_hide_pesos, bias_out, attr_test, rotulos_teste):
    """Evaluate the model's performance on the test set."""
    acertos = 0
    for entrada, rotulo in zip(attr_test, rotulos_teste):
        # Forward: entrada → oculta
        entrada_oculta = [
            bias_oculta[i] + sum(entrada[j] * pesos_in_hide[j][i] for j in range(4))
            for i in range(4)
        ]
        ativacao_oculta = sigmoid(entrada_oculta)

        # Forward: oculta → saída
        entrada_saida = [
            bias_out[i] + sum(ativacao_oculta[j] * out_hide_pesos[j][i] for j in range(4))
            for i in range(3)
        ]
        saida = softmax(entrada_saida)

        pred = saida.index(max(saida))
        if pred == rotulo:
            acertos += 1

    accuracy = acertos / len(rotulos_teste)

    return accuracy, bias_oculta, bias_out

def softmax(x):
    """Calculate the softmax of a list of values."""
    max_x = max(x)
    e_x = [math.exp(i - max_x) for i in x]
    soma = sum(e_x)
    return [i / soma for i in e_x]

def sigmoid(x):
    """Calculate the sigmoid activation function."""
    return [1 / (1 + math.exp(-v)) for v in x]
