"""Main file for the Iris dataset classification using a neural network."""

import os
import treino
import util
import epochs_genetic
import learning_rate_genetic

fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "model")
path1 = os.path.join(fpath,"pesos_entrada_oculta.csv")
path2 = os.path.join(fpath,"saida_oculta_pesos.csv")


PASTA = "model_data"
resp, flower = [], []
CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def main():
    """Main function to run the Iris dataset classification."""
    if (not os.path.exists(path1)) or (not os.path.exists(path2)):
        dataset = util.download_dados()
        dados_shufle = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        _, _, atributos_treino,rotulos_treino = util.divisao_treino_teste(dados_shufle)
        epochs = epochs_genetic.envolve(0.1)
        learning_rate = learning_rate_genetic.envolve(epochs)
        for _ in range(3):

            epochs = epochs_genetic.envolve(learning_rate)
            learning_rate = learning_rate_genetic.envolve(epochs)

        pesos_entrada_oculta, saida_oculta_pesos, _, _ = treino.treinar(atributos_treino,rotulos_treino,epochs,learning_rate)

        os.makedirs(PASTA, exist_ok=True)
        os.makedirs(fpath, exist_ok=True)

        with open(path1, "w", encoding="utf-8") as f:
            for linha in pesos_entrada_oculta:
                f.write(",".join(str(valor) for valor in linha) + "\n")
        with open(path2, "w", encoding="utf-8") as f:
            for linha in saida_oculta_pesos:
                f.write(",".join(str(valor) for valor in linha) + "\n")
    pesos = []

    with open(path1, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            pesos.append(linha_convertida)
    with open(path2, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            pesos.append(linha_convertida)

    while True:
        a = input("Deseja classificar uma flor? (s/n): ").strip().lower()
        if a != "s":
            print("Saindo do programa.")
            break

        print("Digite os atributos da flor:")
        flower.append(float(input("SepalLengthCm:")))
        flower.append(float(input("SepalWidthCm:")))
        flower.append(float(input("PetalLengthCm:")))
        flower.append(float(input("PetalWidthCm:")))

        for i in range(3):  # 3 classes
            score = 0
            for j in range(4):  # 4 atributos
                score += flower[j] * pesos[j][i]
            resp.append(score)

        print(f"Classe prevista:\n{CLASSES[resp.index(max(resp))]}")

if __name__ == "__main__":
    main()
