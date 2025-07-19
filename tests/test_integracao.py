import sys
import os
from sklearn.metrics import accuracy_score
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import util
import treino


def test_pipeline_treino_e_predicao():

    dataset = util.download_dados()
    atributos_teste, rotulos_teste, atributos_treino, rotulos_treino = util.divisao_treino_teste(dataset)

    pesos_entrada_oculta, saida_oculta_pesos, bias_oculta, bias_saida = treino.treinar(
        atributos_treino, rotulos_treino, epocas=200, taxa_aprendizado=0.05
    )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    X_teste = np.array(atributos_teste, dtype=float)
    y_esperado = np.array(rotulos_teste, dtype=int)

    z_oculta = np.dot(X_teste, pesos_entrada_oculta) + bias_oculta
    a_oculta = sigmoid(z_oculta)

    z_saida = np.dot(a_oculta, saida_oculta_pesos) + bias_saida
    saida = softmax(z_saida)

    y_predito = np.argmax(saida, axis=1)

    assert len(y_predito) == len(y_esperado), "Quantidade de predições não bate com esperado"

    acc = accuracy_score(y_esperado, y_predito)
    assert acc >= 0.8, f"Acurácia abaixo do esperado: {acc:.2f}"