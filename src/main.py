"""MAIN MODULE."""

import treino
import util

def main():
    """MAIN."""
    dataset = util.download_dados()
    atributos_teste,rotulos_teste,atributos_treino,rotulos_treino = util.divisao_treino_teste(dataset)
    treino.treinar(atributos_treino, rotulos_treino, 100)

if __name__ == "__main__":
    main()
