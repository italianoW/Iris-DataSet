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
    train = []
    test = []
    print(tamanho)
        #lista[0], lista[48]
        #lista[49],lista[98])
        #lista[99],lista[148])

    lista.insert(0,lista[0])

    for i in range(3):
        for j in range(33):
            train.append(lista[i*50 + j])
    
    for i in range(3):
        for j in range(17):
            test.append(lista[i*50 + 33 + j])

    return train,test
    
    