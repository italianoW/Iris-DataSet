import treino
import util



if __name__ == "__main__":
    
    dataset = util.download_dados()
    dados_de_treino, dados_de_teste = util.divisao_treino_teste(dataset)
    modelo = treino.treinar(dados_de_treino)
    
