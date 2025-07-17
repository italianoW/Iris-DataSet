"""MAIN MODULE."""

import treino
import util

def main():
    """MAIN."""
    dataset = util.download_dados()
    dados_de_treino, _ = util.divisao_treino_teste(dataset)
    treino.treinar(dados_de_treino, 100)

if __name__ == "__main__":
    main()
