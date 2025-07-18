import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import util

def test_download_dados_retorna_dataframe():
	dados = util.download_dados()

	assert isinstance(dados, pd.DataFrame), "download_dados() não retornou um DataFrame"
	assert not dados.empty, "O DataFrame retornado está vazio"
	assert dados.shape[1] >= 5, f"Esperado pelo menos 5 colunas, mas retornou {dados.shape[1]}"

def test_divisao_treino_teste_tamanho_e_rotulos():
	dados = util.download_dados()
	X_train, y_train, X_test, y_test = util.divisao_treino_teste(dados)

	assert len(X_train) == 99, f"Esperado 99 amostras de treino, mas retornou {len(X_train)}"
	assert len(X_test) == 51, f"Esperado 51 amostras de teste, mas retornou {len(X_test)}"

	rotulos_esperados = {0, 1, 2}
	assert rotulos_esperados.issubset(set(y_train)), "Nem todos os rótulos foram encontrados no treino"
	assert rotulos_esperados.issubset(set(y_test)), "Nem todos os rótulos foram encontrados no teste"
