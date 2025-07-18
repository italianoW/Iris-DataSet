"""Main file for the Iris dataset classification using a neural network."""

import os
import math
import treino
import util
import epochs_genetic
import learning_rate_genetic

fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "model_data", "model")
path1 = os.path.join(fpath,"pesos_entrada_oculta.csv")
path2 = os.path.join(fpath,"saida_oculta_pesos.csv")
path3 = os.path.join(fpath,"bias_oculta.csv")
path4 = os.path.join(fpath,"bias_saida.csv")

resp, flower = [], []
CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def main():
    """Main function to run the Iris dataset classification."""
    if ((not os.path.exists(path1))
        or (not os.path.exists(path2))
        or (not os.path.exists(path3))
        or (not os.path.exists(path4))):

        dataset = util.download_dados()
        dados_shufle = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        _, _, attr_train,rotulos_treino = util.divisao_treino_teste(dados_shufle)
        epochs = epochs_genetic.envolve(0.1)
        learning_rate = learning_rate_genetic.envolve(epochs)
        for _ in range(2):
            epochs = epochs_genetic.envolve(learning_rate)
            learning_rate = learning_rate_genetic.envolve(epochs)

        p_in_hide,out_p,b_hide,b_out=treino.treinar(attr_train,rotulos_treino,epochs,learning_rate)

        os.makedirs(fpath, exist_ok=True)

        with open(path1, "w", encoding="utf-8") as f:
            for linha in p_in_hide:
                f.write(",".join(str(valor) for valor in linha) + "\n")
        with open(path2, "w", encoding="utf-8") as f:
            for linha in out_p:
                f.write(",".join(str(valor) for valor in linha) + "\n")
        with open(path3, "w", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in b_hide))
        with open(path4, "w", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in b_out))

    p_in_hide,out_p,b_hide,b_out = [],[],[],[]

    with open(path1, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            p_in_hide.append(linha_convertida)
    with open(path2, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            out_p.append(linha_convertida)
    with open(path3, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            b_hide.append(linha_convertida)
    with open(path4, "r", encoding="utf-8") as f:
        for linha in f:
            valores = linha.strip().split(",")
            linha_convertida = [float(v) for v in valores]
            b_out.append(linha_convertida)
    b_hide = b_hide[0]
    b_out = b_out[0]

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

        camada_oculta = []
        for i in range(4):
            soma = b_hide[i] + sum(
                flower[j] * p_in_hide[j][i] for j in range(4)
            )
            ativacao = 1 / (1 + math.exp(-soma))  # sigmoid
            camada_oculta.append(ativacao)

        entrada_saida = []
        for i in range(3):
            soma = b_out[i] + sum(
                camada_oculta[j] * out_p[j][i] for j in range(4)
            )
            entrada_saida.append(soma)

        saida = util.softmax(entrada_saida)

        print("Ativações:", saida)
        classe_predita = CLASSES[saida.index(max(saida))]
        print(f"Classe prevista:\n{classe_predita}")

if __name__ == "__main__":
    main()
