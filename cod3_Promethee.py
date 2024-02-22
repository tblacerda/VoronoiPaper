import pandas as pd
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
import warnings
warnings.filterwarnings('ignore')


def Promethee(Entrada, topN):
    '''
    Entrada: Dataframe com os sites a serem avaliados
    as 4 primeiras linhas referem-se aos parametros do Promethee
    peso: peso da feature na decisão. a linha soma 100%
    tipo: 1 ou -1. 1 é maior melhor. -1 menor melhor
    p: parametro de ajuste do promethee para folga. Se nao souber, deixar zero.
    q:parametro de ajuste do promethee para rampa. Se nao souber, deixar zero.

    '''
    df = pd.read_excel(Entrada)
    # Extrai os pesos das colunas a partir da primeira linha do dataframe
    weights = np.array(df.iloc[0, 1:].astype(float))
    # Extrai o tipo de cada coluna a partir da segunda linha do dataframe
    types = np.array(df.iloc[1, 1:].astype(float))
    # Extrai o valor do parâmetro p do método PROMETHEE II a partir da
    # terceira linha
    p = np.array(df.iloc[2, 1:].astype(float))
    # Extrai o valor do parâmetro q do método PROMETHEE II a partir da
    # quarta linha
    q = np.array(df.iloc[3, 1:].astype(float))
    # Extrai os valores de decisão para cada alternativa a partir da
    # quinta linha do dataframe
    valores = np.array(df.iloc[4:, 1:].astype(float))
    # Cria um objeto PROMETHEE_II com 'vshape_2' como argumento
    pref = PROMETHEE_II('vshape_2')
    # Calcula a preferência para cada alternativa com base nos
    # pesos, tipos, 'p' e 'q'
    pref = pref(valores, weights, types, q=q, p=p)
    # Calcula o ranking das alternativas com base nas preferências
    ranking = rrankdata(pref)
    # Remove as primeiras quatro linhas do dataframe,
    # que contêm os pesos, tipos, 'p' e 'q'
    df = df.iloc[4:, ]
    # Cria uma ultima coluna chamada Ranking
    df['ranking'] = ranking
    # Ordena do mais provavel para o menos provavel.
    df = df.sort_values(by=['ranking'], ascending=True)
    # corta as topN
    df = df.iloc[0:topN:]
    # Retorna a tabela
    return df

Promethee('entrada.xlsx',3)
