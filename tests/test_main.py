# import sys
# import os
# import builtins
# import shutil
# from unittest.mock import patch, MagicMock, mock_open
# import pytest

# # Adiciona o diretório src ao path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import main

# PASTA = os.path.join(os.path.dirname(__file__), '..', 'model_data')
# FPATH = os.path.join(PASTA, 'pesos.csv')

# # Simulação de entrada para classificação
# ENTRADAS_FAKE = iter(["5.1", "3.5", "1.4", "0.2"])

# @pytest.fixture(autouse=True)
# def limpar_modelo():
# 	# Remove o arquivo de pesos antes e depois dos testes
# 	if os.path.exists(FPATH):
# 		os.remove(FPATH)
# 	yield
# 	if os.path.exists(FPATH):
# 		os.remove(FPATH)

# @pytest.fixture
# def simula_input():
# 	with patch("builtins.input", lambda: next(ENTRADAS_FAKE)):
# 		yield

# def test_main_gera_modelo_e_classifica(monkeypatch, capsys, simula_input):
# 	# Simula a execução da main sem arquivo de pesos (caminho "if not os.path.exists(...)")
# 	try:
# 		main.main()
# 	except StopIteration:
# 		pass

# 	# Verifica se o arquivo foi gerado
# 	assert os.path.exists(FPATH), "O arquivo de pesos.csv deveria ter sido criado"

# 	# Verifica se a classificação foi impressa
# 	saida = capsys.readouterr()
# 	assert "Classe prevista:" in saida.out

# def test_main_utiliza_modelo_existente(monkeypatch, capsys, simula_input):
# 	# Cria um arquivo de pesos manualmente para forçar o caminho "else"
# 	os.makedirs(PASTA, exist_ok=True)
# 	with open(FPATH, "w") as f:
# 		f.write("0.1,0.2,0.3,0.4\n0.5,0.6,0.7,0.8\n0.9,1.0,1.1,1.2\n")

# 	# Executa a main
# 	try:
# 		main.main()
# 	except StopIteration:
# 		pass

# 	# Verifica se o arquivo foi reutilizado e a classificação impressa
# 	saida = capsys.readouterr()
# 	assert "Classe prevista:" in saida.out

# def test_main_roda_sem_excecao(monkeypatch, simula_input):
# 	try:
# 		main.main()
# 	except Exception as e:
# 		pytest.fail(f"main.main() levantou exceção inesperada: {e}")
