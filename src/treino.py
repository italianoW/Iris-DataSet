"""TREINO MODULE."""
import util
import math

def treinar(atributos, rotulos, epocas, taxa_aprendizado):
    """TREINAR."""
    tamanho = len(rotulos)

    def one_hot(i, tamanho):
        """ONE-HOT."""
        vetor = [0] * tamanho
        vetor[i] = 1
        return vetor
    
    def softmax(x):
        e_x = [math.exp(i) for i in x]
        soma = sum(e_x)
        return [i / soma for i in e_x]

    rotulos_onehot = [one_hot(i, 3) for i in rotulos]

    pesos = [[ 0.1 for _ in range(3)] for _ in range(4)]
    bias = [0.0, 0.0, 0.0]
    

# === Treinamento ===

    erro_medio_final = -1

    for epoca in range(epocas):

        for entrada, esperado in zip(atributos, rotulos_onehot):
        # Feedforward (sem função de ativação)
            saida = softmax([bias[i] + sum(entrada[j] * pesos[j][i] 
                                   for j in range(4)) 
                     for i in range(3)]) #avalia com qual petala mais se parece conforme os pesos


        # Backpropagation (ajuste dos pesos)
            for i in range(3):  # para cada neurônio de saída
            
                erro = saida[i] - esperado[i]                         #Ex: Classe correta é Versicolor, então o esperado é [0, 1, 0]
                                                                  #A saída da rede foi [0.3, 0.5, 0.2]
                                                                  #erro_classe_0 = 0.3 - 0 = +0.3 Vai reduzir [0] pois ativou sem motivo
                                                                  #erro_classe_1 = 0.5 - 1 = -0.5 Vai aumentar [1] pois ativou menos do que necessário
                                                                  #erro_classe_2 = 0.2 - 0 = +0.2 Vai reduzir [2] pois ativou sem motivo
                for j in range(4):  # para cada entrada
                    pesos[j][i] -= taxa_aprendizado * erro * entrada[j]
                                                                  #Ex:Erro = +0.3 (a rede ativou mais do que deveria),
                                                                  #entrada = [2, 0, 1, 4],
                                                                  #taxa_aprendizado = 0.1
                                                                  #Para cada atributo j, o peso será atualizado assim:
                                                                  #pesos[0][i] -= 0.1 * 0.3 * 2  → -0.06
                                                                  #pesos[1][i] -= 0.1 * 0.3 * 0  → 0     (não muda)
                                                                  #pesos[2][i] -= 0.1 * 0.3 * 1  → -0.03
                                                                  #pesos[3][i] -= 0.1 * 0.3 * 4  → -0.12
                bias[i] -= taxa_aprendizado * erro

                #def isNaNlist(lista):
                #    for n in lista:
                #        if math.isnan(n):
                #            return False
                #    return True
                #def isNaNmatriz(matriz):
                 #   for n in matriz:
                 #       for m in n:
                 #           if math.isnan(m):
                 #               return False
                 #   return True
                    
                #if isNaNlist(bias) and isNaNmatriz(pesos):
                #    print(pesos ,bias)
                            
    #print(bias)
    return pesos, bias
