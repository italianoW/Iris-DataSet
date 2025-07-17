import pandas as pd
import kagglehub
import os


def download_dados(): #returns data from iris_dataset 

    path = kagglehub.dataset_download("vikrishnan/iris-dataset")
    print(os.listdir(path))
    return pd.read_csv(os.path.join(path, 'iris.data.csv'))

def divisao_treino_teste(dados):

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

    tamanho = len(rotulos_teste)
    
    for i in range(tamanho):
        saida = [bias[i] + sum(atributos_teste[j] * modelo[j][i] for j in range(4))for i in range(3)]
        erro_total += mse(saida, rotulos_teste[i])

    return erro_total, bias


    