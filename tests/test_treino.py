import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import treino


 
def test_treinar_retorna_pesos_com_tamanho_correto():
	exemplo_dados = [
		[5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
		[7.0, 3.2, 4.7, 1.4, "Iris-versicolor"],
		[6.3, 3.3, 6.0, 2.5, "Iris-virginica"],
		[5.0, 3.6, 1.4, 0.2, "Iris-setosa"],
		[6.7, 3.1, 4.4, 1.4, "Iris-versicolor"],
	]

	pesos = treino.treinar(exemplo_dados)

	assert isinstance(pesos, list), "Retorno deve ser uma lista"
	assert len(pesos) == 4, f"Deve ter 4 listas internas (uma para cada atributo), tem {len(pesos)}"
	assert all(len(p) == 3 for p in pesos), "Cada lista interna deve ter 3 pesos (uma por saída)"

def test_treinar_pesos_variam():
	exemplo_dados = [
		[5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
		[7.0, 3.2, 4.7, 1.4, "Iris-versicolor"],
	]

	pesos_antes = [[0.1 for _ in range(3)] for _ in range(4)]

	pesos_depois = treino.treinar(exemplo_dados)

	assert pesos_depois != pesos_antes, "Os pesos deveriam ter sido atualizados no treino"