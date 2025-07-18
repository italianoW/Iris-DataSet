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
 
def test_divisao_treino_teste_tamanho_e_divisao():
	dados = util.download_dados()

	train = util.divisao_treino_teste(dados)

	assert len(train) == 4, f"Treino deveria ter 99 amostras, tem {len(train)}"