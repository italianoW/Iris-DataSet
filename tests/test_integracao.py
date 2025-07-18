import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import util
import treino

def test_pipeline_treino_e_predicao():
    dataset = util.download_dados()
    treino_data, teste_data = util.divisao_treino_teste(dataset)
    modelo = treino.treinar(treino_data)

    X_teste = teste_data.drop(columns='target')
    y_esperado = teste_data['target']
    y_predito = modelo.predict(X_teste)

    assert len(y_predito) == len(y_esperado), "Número de predições diferente do esperado"
    acc = accuracy_score(y_esperado, y_predito)
    assert acc >= 0.8, f"Acurácia abaixo do esperado: {acc:.2f}"