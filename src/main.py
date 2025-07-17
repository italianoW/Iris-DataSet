"""MAIN MODULE."""

import treino
import util
import epochs_genetic
import learning_rate_genetic
import os
import numpy as np

def main():
    caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "pesos.csv")
    if not os.path.exists(caminho_arquivo):
        """MAIN."""
        dataset = util.download_dados()
        _,_,atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
        epochs = epochs_genetic.learn(0.1)
        learning_rate = learning_rate_genetic.learn(epochs)
        for _ in range(3):
            epochs = epochs_genetic.learn(learning_rate)
            learning_rate = learning_rate_genetic.learn(epochs)

        pesos,_ = treino.treinar(atributos_treino,rotulos_treino,epochs,learning_rate)
        # Pasta para salvar
        pasta = "model_data"
        os.makedirs(pasta, exist_ok=True)

        # Salvar os valores no arquivo (sem cabeçalho)
        with open(caminho_arquivo, "w") as f:
            for linha in pesos:
                linha_str = ",".join(str(valor) for valor in linha)
                f.write(linha_str + "\n")

    pesos = []
    
    with open(caminho_arquivo, "r") as f:
        for linha in f:
            valores = linha.strip().split(",")      # Remove o \n e divide pelos ','
            linha_convertida = [float(v) for v in valores]  # Converte cada string para float
            pesos.append(linha_convertida)

    while True:
        print("SepalLengthCm:")
        sl = float(input()) 
        print("SepalWidthCm:")
        sw = float(input())
        print("PetalLengthCm:")
        pl = float(input())
        print("PetalWidthCm:")
        pw = float(input())
        print("Species:")
        
        flower = (sl, sw, pl, pw)
        
        resp = []
        for i in range(3):  # 3 classes
            score = 0
            for j in range(4):  # 4 atributos
                score += flower[j] * pesos[j][i]
            resp.append(score)

        classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        print("Classe prevista:", classes[resp.index(max(resp))])
             

if __name__ == "__main__":
    main()
