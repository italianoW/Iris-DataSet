import pytest
import pandas as pd
import  sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(src_path))

from iris_dataset import mse, one_hot

# TESTES DE INICIALIZAÇÃO DOS DADOS

@pytest.fixture
def dados_amostrais():
    return [
        [5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
        [7.0, 3.2, 4.7, 1.4, "Iris-versicolor"],
        [6.3, 3.3, 6.0, 2.5, "Iris-virginica"]
    ]

def test_carregamento_dados(dados_amostrais):
    df = pd.DataFrame(dados_amostrais, columns=['f1', 'f2', 'f3', 'f4', 'classe'])
    lista = df.values.tolist()

    assert len(lista) == 3
    assert lista[0][:4] == [5.1, 3.5, 1.4, 0.2]
    assert lista[1][4] == "Iris-versicolor"

def test_conversao(dados_amostrais):

    rotulos = []

    for linha in dados_amostrais:
        rotulos.append(0 if linha[4] == "Iris-setosa" else
                       1 if linha[4] == "Iris-versicolor" else 2)
    
    assert rotulos == [0, 1, 2]
    assert rotulos[2] == 2


# TESTES ONE_HOT

def test_one_hot():
    assert one_hot(0, 3) == [1, 0, 0]
    assert one_hot(1, 3) == [0, 1, 0]
    assert one_hot(2, 3) == [0, 0, 1]

def test_one_hot_tamanhos_diferentes():
    assert one_hot(0, 4) == [1, 0, 0, 0]
    assert one_hot(3, 5) == [0, 0, 0, 1, 0]
    assert one_hot(0, 1) == [1]

def test_one_hot_ultimas_posicoes():
    assert one_hot(2, 3) == [0, 0, 1]
    assert one_hot(3, 4) == [0, 0, 0, 1]
    assert one_hot(5, 6) == [0, 0, 0, 0, 0, 1]

def test_one_hot_primeiras_posicoes():
    assert one_hot(0, 3) == [1, 0, 0]
    assert one_hot(0, 4) == [1, 0, 0, 0]
    assert one_hot(0, 6) == [1, 0, 0, 0, 0, 0]   

def test_one_hot_posicao_invalida():
    
    with pytest.raises(IndexError):
        one_hot(3, 3)
    
    with pytest.raises(IndexError):
        one_hot(-1, 3)

def test_one_hot_tamanho_invalido():
    
    with pytest.raises(ValueError):
        one_hot(0, 0)

    with pytest.raises(ValueError):
        one_hot(0, -1)


# TESTES MSE (ERRO PADRÃO MÉDIO)

def test_mse_lista_vazia():
    with pytest.raises(ZeroDivisionError):
        mse([],[])

def test_mse_lista_tamanhos_diferentes():
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])

def test_mse_predicao_zero():
    val_verdadeiros = [1.0, 2.0, 3.0]
    val_previstos = [0.0, 0.0, 0.0]
    esperado = (1 + 4 + 9) / 3
    assert mse(val_verdadeiros, val_previstos) == pytest.approx(esperado)

def test_mse_erro_constante():
    val_verdadeiros = [1.0, 2.0, 3.0]
    val_previstos = [1.5, 2.5, 3.5]
    esperado = 3 * (0.5**2) / 3
    assert mse(val_verdadeiros, val_previstos) == pytest.approx(esperado)

def test_mse_erros_grandes():
    val_verdadeiros = [1.0, 2.0, 3.0]
    val_previstos = [10.0, 5.0, 20.0]
    esperado = ((9**2) + (3**2) + (17**2)) / 3
    assert mse(val_verdadeiros, val_previstos) == pytest.approx(esperado)