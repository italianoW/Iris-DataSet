import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import util
import treino

def test_predicao_valida():
    dados = util.download_dados()
    atributos_teste,rotulos_teste,atributos_treino,rotulos_treino = util.divisao_treino_teste(dados)

    pesos, bias = treino.treinar(atributos_treino, rotulos_treino, epocas=10, taxa_aprendizado=0.05)

    accuracy, _ = util.avaliar(bias, pesos, atributos_teste, rotulos_teste)

    assert 0 <= accuracy <= 1, "Accuracy deve estar entre 0 e 1"
    assert accuracy > 0.5, "Accuracy deve ser maior que 0.5 para ser razoável"
