# import sys
# import os
# from sklearn.metrics import f1_score

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# import util
# import treino

# def test_f1_score_classificador():
#     dataset = util.download_dados()
#     atributos_teste, rotulos_teste, atributos_treino, rotulos_treino = util.divisao_treino_teste(dataset)
    
#     pesos, bias = treino.treinar(atributos_treino, rotulos_treino, epocas=10, taxa_aprendizado=0.05)

#     y_pred = util.avaliar(bias, pesos, atributos_teste, rotulos_teste)

#     f1 = f1_score(atributos_teste, y_pred, average='macro')

#     assert 0 <= f1 <= 1
#     assert f1 >= 0.8, f"F1-score abaixo do esperado: {f1:.2f}"