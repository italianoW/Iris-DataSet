import pandas as pd
import os
import math
import random

def download_dados(): #returns data from iris_dataset 

    return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "iris.data.csv"))

def divisao_treino_teste(dados):

    random.seed(42)    
    lista = dados.values.tolist()
    tamanho = len(lista)
    treino,teste = [], []
    atributos_teste, rotulos_teste, atributos_treino, rotulos_treino = [], [], [], []

        #lista[0], lista[48]
        #lista[49],lista[98])
        #lista[99],lista[148])

    lista.insert(0,lista[0])

    for i in range(3):
        for j in range(33):
            treino.append(lista[i*50 + j])
    
    for i in range(3):
        for j in range(17):
            teste.append(lista[i*50 + 33 + j])

    random.shuffle(treino)
    random.shuffle(teste)

    for i in range(99):
        atributos_treino.append(treino[i][:4])
        rotulos_treino.append(0 if lista[i][4]== "Iris-setosa" else 1 if lista[i][4]== "Iris-versicolor" else 2) 

    for i in range(51):
        atributos_teste.append(treino[i][:4])
        rotulos_teste.append(0 if lista[i][4]== "Iris-setosa" else 1 if lista[i][4]== "Iris-versicolor" else 2) 

    return atributos_teste,rotulos_teste,atributos_treino,rotulos_treino

def mse(saida, esperado):
        """MSE."""
        return sum((s - e) ** 2 for s, e in zip(saida, esperado)) / len(saida)

def avaliar(bias, modelo, atributos_teste, rotulos_teste):
    acertos = 0
    for atributos, rotulo in zip(atributos_teste, rotulos_teste):
        saida = relu([bias[i] + sum(atributos[j] * modelo[j][i] for j in range(4)) for i in range(3)])
        saida = softmax(saida)
        pred = saida.index(max(saida))
        if pred == rotulo:
            acertos += 1
    accuracy = acertos / len(rotulos_teste)
    return accuracy, bias

def softmax(x):
    max_x = max(x)
    e_x = [math.exp(i - max_x) for i in x]  # shift to prevent overflow
    soma = sum(e_x)
    return [i / soma for i in e_x]

def relu(x):
    return [max(0,v) for v in x]