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

    pesos, _ = treino.treinar(atributos, rotulos, 10, 0.01)
    assert isinstance(pesos, list), "Deve retornar uma lista de pesos"
    assert len(pesos) == 4, f"Deve ter 4 listas internas (uma para cada atributo), tem {len(pesos)}"
    assert all(len(p) == 3 for p in pesos), "Cada lista interna deve ter 3 pesos (uma por saída)"


def test_treinar_pesos_variam():
	atributos = [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],]
      
	rotulos = [0, 2, 1, 2]

	pesos_antes = [[0.1 for _ in range(3)] for _ in range(4)]

	pesos_depois = treino.treinar(atributos, rotulos, 10, 0.01)

	# Como seu treino sempre começa com pesos fixos 0.1, espera-se que pesos_depois != pesos_antes
	assert pesos_depois != pesos_antes, "Os pesos deveriam ter sido atualizados no treino"