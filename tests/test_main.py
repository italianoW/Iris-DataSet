import sys
import os
import builtins
import shutil
from unittest.mock import patch, MagicMock
import pytest
import subprocess

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import main

PASTA = os.path.join(os.path.dirname(__file__), '..', 'model_data')
FPATH = os.path.join(PASTA, 'pesos.csv')

@pytest.fixture(autouse=True)
def limpar_modelo():
	# Remove o arquivo de pesos antes e depois dos testes
	if os.path.exists(FPATH):
		os.remove(FPATH)
	yield
	if os.path.exists(FPATH):
		os.remove(FPATH)

def mock_input_generator(*respostas):
	respostas_iter = iter(respostas)
	return lambda _: next(respostas_iter)

def test_main_gera_modelo_e_classifica(monkeypatch, capsys):
	# Simula entradas: deseja classificar -> s -> atributos -> n (sai)
	entradas = ["s", "5.1", "3.5", "1.4", "0.2", "n"]
	monkeypatch.setattr("builtins.input", mock_input_generator(*entradas))

	main.main()

	# Verifica se o arquivo foi gerado
	assert os.path.exists(FPATH), "O arquivo de pesos.csv deveria ter sido criado"

	# Verifica se a classificação foi impressa
	saida = capsys.readouterr()
	assert "Classe prevista:" in saida.out

def test_main_utiliza_modelo_existente(monkeypatch, capsys):
    # Cria um arquivo de pesos manualmente para forçar o caminho "else"
    os.makedirs(PASTA, exist_ok=True)
    with open(FPATH, "w") as f:
        f.write("0.1,0.2,0.3\n0.4,0.5,0.6\n0.7,0.8,0.9\n0.9,1.0,1.1\n")

    entradas = ["s", "5.1", "3.5", "1.4", "0.2", "n"]
    monkeypatch.setattr("builtins.input", mock_input_generator(*entradas))

    main.main()

    saida = capsys.readouterr()
    assert "Classe prevista:" in saida.out

def test_main_roda_sem_excecao(monkeypatch):
	entradas = ["s", "5.1", "3.5", "1.4", "0.2", "n"]
	monkeypatch.setattr("builtins.input", mock_input_generator(*entradas))

	try:
		main.main()
	except Exception as e:
		pytest.fail(f"main.main() levantou exceção inesperada: {e}")
  
def test_main_py_executa_como_script():
	entradas = "\n".join(["s", "5.1", "3.5", "1.4", "0.2", "n"]) + "\n"

	os.makedirs(PASTA, exist_ok=True)
	with open(FPATH, "w") as f:
		f.write("0.1,0.2,0.3\n0.4,0.5,0.6\n0.7,0.8,0.9\n0.9,1.0,1.1\n")

	main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'))

	result = subprocess.run(
		[sys.executable, main_path],
		input=entradas,
		capture_output=True,
		text=True
	)

	assert result.returncode == 0, f"main.py retornou erro: {result.stderr}"
