"""This file contains the training function for a neural network model"""
import math
import random
import util as u

random.seed(42)

LIMITE_SUP = 4
LIMITE_INF = -4


def treinar(atributos, rotulos, epocas, taxa_aprendizado):
    """Train a neural network model using the provided attributes and labels."""
    pesos_entrada_oculta = [[random.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(4)]  # 4x4
    bias_oculta = [random.uniform(-0.1, 0.1) for _ in range(4)]

    saida_oculta_pesos = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(4)]    # 4x3
    bias_saida = [random.uniform(-0.1, 0.1) for _ in range(3)]



    def one_hot(i, tamanho):
        """Convert a label index into a one-hot encoded vector."""
        vetor = [0] * tamanho
        vetor[i] = 1
        return vetor

    def softmax(x):
        """Calculate the softmax of a list of values."""
        max_x = max(x)
        e_x = [math.exp(i - max_x) for i in x]
        soma = sum(e_x)
        return [i / soma for i in e_x]

    rotulos_onehot = [one_hot(i, 3) for i in rotulos]


# === Treinamento ===

    for _ in range(epocas):

        for entrada, esperado in zip(atributos, rotulos_onehot):
            
            # Cálculo da ativação da camada oculta
            entrada_oculta = [bias_oculta[i] + sum(entrada[j] * pesos_entrada_oculta[j][i] for j in range(4))
                                                                                           for i in range(4)]
            ativacao_oculta = u.sigmoid(entrada_oculta)

        # Cálculo da ativação da camada de saída
            entrada_saida = [
            bias_saida[i] + sum(ativacao_oculta[j] * saida_oculta_pesos[j][i] for j in range(4))
                                                                              for i in range(3)]
            saida = softmax(entrada_saida)

            erro_saida = [saida[i] - esperado[i] for i in range(3)]


            for i in range(3):
                for j in range(4):

                    saida_oculta_pesos[j][i] -= taxa_aprendizado * erro_saida[i] * ativacao_oculta[j]
                    saida_oculta_pesos[j][i] = max(min(saida_oculta_pesos[j][i], LIMITE_SUP), LIMITE_INF)

                bias_saida[i] -= taxa_aprendizado * erro_saida[i]
                bias_saida[i] = max(min(bias_saida[i], LIMITE_SUP), LIMITE_INF)

            sigmoid_oculta = u.sigmoid(entrada_oculta)
            sigmoid_oculta = [s * (1 - s) for s in sigmoid_oculta]
            erro_oculta = [sigmoid_oculta[i] * sum(erro_saida[k] * saida_oculta_pesos[i][k] for k in range(3))
                                                                                            for i in range(4)]

# Atualização dos pesos da oculta
            for j in range(4):
                for k in range(4):
                    pesos_entrada_oculta[j][k] -= taxa_aprendizado * erro_oculta[k] * entrada[j]
                    pesos_entrada_oculta[j][k] = max(min(pesos_entrada_oculta[j][k], LIMITE_SUP), LIMITE_INF)

            for k in range(4):
                bias_oculta[k] -= taxa_aprendizado * erro_oculta[k]
                bias_oculta[k] = max(min(bias_oculta[k], LIMITE_SUP), LIMITE_INF)
                
    
    return pesos_entrada_oculta, saida_oculta_pesos, bias_oculta, bias_saida
