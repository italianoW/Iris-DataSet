"""Main file for the Iris dataset classification using a neural network."""

import os
import treino
import util
import epochs_genetic
import learning_rate_genetic

fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "pesos.csv")
PASTA = "model_data"
resp, flower = [], []
CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def main():
    """Main function to run the Iris dataset classification."""
    if not os.path.exists(fpath):
        dataset = util.download_dados()
        _, _, atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
        epochs = epochs_genetic.envolve(0.1)
        learning_rate = learning_rate_genetic.envolve(epochs)
        for _ in range(3):

            epochs = epochs_genetic.envolve(learning_rate)
            learning_rate = learning_rate_genetic.envolve(epochs)

        pesos, _ = treino.treinar(atributos_treino,rotulos_treino,epochs,learning_rate)

        os.makedirs(PASTA, exist_ok=True)

        with open(fpath, "w", encoding="utf-8") as f:
            for linha in pesos:
                linha_str = ",".join(str(valor) for valor in linha)
                f.write(linha_str + "\n")

    pesos = []

    with open(fpath, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            pesos.append(linha_convertida)

    while True:
        print("SepalLengthCm:")
        flower.append(float(input()))
        print("SepalWidthCm:")
        flower.append(float(input()))
        print("PetalLengthCm:")
        flower.append(float(input()))
        print("PetalWidthCm:")
        flower.append(float(input()))

        for i in range(3):  # 3 classes
            score = 0
            for j in range(4):  # 4 atributos
                score += flower[j] * pesos[j][i]
            resp.append(score)

        print("Classe prevista:", CLASSES[resp.index(max(resp))])

if __name__ == "__main__":
    main()
