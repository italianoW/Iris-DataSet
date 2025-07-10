import pandas as pd
import kagglehub
import os
import random

# Download latest version
path = kagglehub.dataset_download("vikrishnan/iris-dataset")

print(os.listdir(path))

dados = pd.read_csv(os.path.join(path, 'iris.data.csv'))

lista = dados.values.tolist() 

atributos = []
rotulos = []
for i in range(149):
  atributos.append(lista[i][:4])
  rotulos.append(0 if lista[i][4]== "Iris-setosa" else 1 if lista[i][4]== "Iris-versicolor" else 2) 

nomes_flores = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# === One-hot manual ===
def one_hot(i, tamanho):
    vetor = [0] * tamanho #Cria um vetor de valor para indicar tipo de flor
    vetor[i] = 1      #(relacionado a tabela entrada)
    return vetor

rotulos_onehot = [one_hot(i, 3) for i in rotulos]

pesos = [[ 0.1 for _ in range(3)] for _ in range(4)]  # 4 entradas → 3 saídas
bias = [0.0, 0.0, 0.0]

# Erro quadrático médio (MSE)
def mse(saida, esperado):
    return sum((s - e) ** 2 for s, e in zip(saida, esperado)) / len(saida)

# === Treinamento ===
taxa_aprendizado = 0.01
epocas = 101

for epoca in range(epocas):
    erro_total = 0
    for entrada, esperado in zip(atributos, rotulos_onehot):
        # Feedforward (sem função de ativação)
        saida = [bias[i] + sum(entrada[j] * pesos[j][i] for j in range(4)) for i in range(3)] #avalia com qual petala mais se parece conforme os pesos 
                                                                                              #previamente dados, dando valores menores para petalas que  
                                                                                              #parecem menos ex: (-0.9, 3, 1.7).
        

        # Calcula erro e acumula perda
        erro_total += mse(saida, esperado)  

        # Backpropagation (ajuste dos pesos)
        for i in range(3):  # para cada neurônio de saída
            
            erro = saida[i] - esperado[i]                         #Ex: Classe correta é Versicolor, então o esperado é [0, 1, 0]
                                                                  #A saída da rede foi [0.3, 0.5, 0.2]
                                                                  #erro_classe_0 = 0.3 - 0 = +0.3 Vai reduzir [0] pois ativou sem motivo
                                                                  #erro_classe_1 = 0.5 - 1 = -0.5 Vai aumentar [1] pois ativou menos do que necessário
                                                                  #erro_classe_2 = 0.2 - 0 = +0.2 Vai reduzir [2] pois ativou sem motivo
            for j in range(4):  # para cada entrada
                pesos[j][i] -= taxa_aprendizado * erro * entrada[j]
                                                                  #Ex:Erro = +0.3 (a rede ativou mais do que deveria),
                                                                  #entrada = [2, 0, 1, 4],
                                                                  #taxa_aprendizado = 0.1
                                                                  #Para cada atributo j, o peso será atualizado assim:
                                                                  #pesos[0][i] -= 0.1 * 0.3 * 2  → -0.06
                                                                  #pesos[1][i] -= 0.1 * 0.3 * 0  → 0     (não muda)
                                                                  #pesos[2][i] -= 0.1 * 0.3 * 1  → -0.03
                                                                  #pesos[3][i] -= 0.1 * 0.3 * 4  → -0.12
            bias[i] -= taxa_aprendizado * erro

    if epoca % 50 == 0:
        print(f"Época {epoca} | Erro médio: {erro_total / len(atributos):.4f}")

