import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import main

def test_main_executa_sem_erro():
    try:
        main.main()
    except Exception as e:
        assert False, f"A função main() levantou exceção: {e}"

def test_main_py_executa_como_script():
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'))
    result = subprocess.run([sys.executable, main_path], capture_output=True, text=True)

    assert result.returncode == 0, f"main.py retornou erro: {result.stderr}"
