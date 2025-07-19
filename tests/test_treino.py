import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import treino

def test_treinar_retorna_pesos_com_tamanho_correto():
    atributos = [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
        [5.0, 3.6, 1.4, 0.2],
    ]
    rotulos = [0, 2, 1, 2]

    pesos_entrada_oculta, saida_oculta_pesos, bias_oculta, bias_saida = treino.treinar(atributos, rotulos, 10, 0.01)

    assert isinstance(pesos_entrada_oculta, list), "pesos_entrada_oculta deve ser uma lista"
    assert len(pesos_entrada_oculta) == 4, "Deve haver um vetor de pesos para cada atributo de entrada"

    num_ocultas = len(pesos_entrada_oculta[0])
    assert all(len(p) == num_ocultas for p in pesos_entrada_oculta), "Todos os vetores de pesos devem ter o mesmo tamanho (número de neurônios ocultos)"

    assert isinstance(saida_oculta_pesos, list), "saida_oculta_pesos deve ser uma lista"
    assert len(saida_oculta_pesos) == num_ocultas, "Deve haver um vetor de pesos para cada neurônio oculto"
    assert all(len(p) == 3 for p in saida_oculta_pesos), "Cada vetor de pesos da camada oculta deve ter 3 elementos (uma para cada saída)"

    assert len(bias_oculta) == num_ocultas, "Bias da camada oculta deve ter mesmo tamanho da camada oculta"
    assert len(bias_saida) == 3, "Bias da camada de saída deve ter 3 elementos (um para cada classe)"


def test_treinar_pesos_variam():
	atributos = [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],]
      
	rotulos = [0, 2, 1, 2]

	pesos_antes = [[0.1 for _ in range(3)] for _ in range(4)]

	pesos_depois = treino.treinar(atributos, rotulos, 10, 0.01)

	assert pesos_depois != pesos_antes, "Os pesos deveriam ter sido atualizados no treino"