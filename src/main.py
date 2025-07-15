import treino
import util

def main():
    dataset = util.download_dados()
    dados_de_treino, dados_de_teste = util.divisao_treino_teste(dataset)
    modelo = treino.treinar(dados_de_treino)

if __name__ == "__main__":
    main()