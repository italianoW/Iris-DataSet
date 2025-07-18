import sys
import os
from sklearn.metrics import f1_score
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import util
import treino


def test_f1_score_modelo():
    dataset = util.download_dados()
    atributos_teste, rotulos_teste, atributos_treino, rotulos_treino = util.divisao_treino_teste(dataset)

    pesos, bias = treino.treinar(atributos_treino, rotulos_treino, epocas=200, taxa_aprendizado=0.05)

    X_teste = np.array(atributos_teste, dtype=float)
    y_esperado = np.array(rotulos_teste, dtype=int)
    pesos = np.array(pesos, dtype=float)
    bias = np.array(bias, dtype=float)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    saida = sigmoid(np.dot(X_teste, pesos) + bias)
    y_predito = np.argmax(saida, axis=1)

    assert len(y_predito) == len(y_esperado), "Quantidade de predições não bate com esperado"

    f1 = f1_score(y_esperado, y_predito, average='macro')
    assert f1 >= 0.8, f"F1-Score abaixo do esperado: {f1:.2f}"
