import sys
import os
import subprocess
import builtins
from unittest.mock import patch, mock_open, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import main

# def test_main_executa_sem_erro():
#     try:
#         main.main()
#     except Exception as e:
#         assert False, f"A função main() levantou exceção: {e}"

# def test_main_py_executa_como_script():
#     main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'))
#     result = subprocess.run([sys.executable, main_path], capture_output=True, text=True)

#     assert result.returncode == 0, f"main.py retornou erro: {result.stderr}"

def test_modelo_é_treinado_ou_arquivo_existente(monkeypatch):
    inputs = iter(["5.1", "3.5", "1.4", "0.2"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    try:
        main.main()
    except StopIteration:
        pass

    caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_data", "pesos.csv")
    assert os.path.exists(caminho_arquivo)

def test_classificacao_simples(monkeypatch, capsys):
    inputs = iter(["5.1", "3.5", "1.4", "0.2"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    try:
        main.main()
    except StopIteration:
        pass
    except KeyboardInterrupt:
        pass

    captured = capsys.readouterr()
    assert "Classe prevista:" in captured.out