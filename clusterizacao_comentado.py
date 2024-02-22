# Python 3.10.12
# Importações necessárias
import pandas as pd #2.0.3
import numpy as np #1.25.2
from sklearn.cluster import KMeans #1.3.0
from datetime import datetime, timedelta 

#31/07 a 27/08
# Definindo as constantes relacionadas às datas
_DIAS_INICIO_ = 0  # Para evitar perturbações de reprocessamentos, #
                   # consideramos D-2 para trás
_JANELA_DIAS_ = 40 + _DIAS_INICIO_  # JANELA DE DIAS               #
_HOJE_ = datetime.today().date()  # Obtemos a data atual
_HOJE_STRING_ = _HOJE_.strftime('%Y-%m-%d')  # Convertendo a data atual para string
_D_2_ = _HOJE_ - timedelta(days=_DIAS_INICIO_)  # Calculamos a data há D-2 dias
_D_2_STRING_ = _D_2_.strftime('%Y-%m-%d')  # Convertendo a data D-2 para string
_D_JANELA_ = _HOJE_ - timedelta(days=_JANELA_DIAS_)  # Calculamos a data há JANELA_DIAS dias
_D_JANELA_STRING_ = _D_JANELA_.strftime('%Y-%m-%d')  # Convertendo a data da janela para string
_ANO_MES_ = _HOJE_.strftime("%Y%m")
__MAX_CLUSTERS__ = 31

# Nome da coluna de data
_COLUNA_DATA_ = 'DIA'

# Chave para o algoritmo K-means
_CHAVE_KMEANS_ = 'REGIONAL'

#202308_PE_ABREU E LIMA_0
# coluna nova com o anomes de execucao 202308


def LerDados(dadosEcq, dadosCidades):
    '''
    Lê e processa os dados dos arquivos de entrada.

    Parâmetros:
        - dadosEcq (str): Caminho para o arquivo de dados ECQ.
        - dadosCidades (str): Caminho para o arquivo de classificação das cidades.

    Retorna:
        - df (DataFrame): DataFrame dos dados ECQ processados.
        - dfCidades (DataFrame): DataFrame de classificação das cidades.
    '''

    # Lê o arquivo de classificação das cidades
    dfCidades = pd.read_excel(dadosCidades)
    
    # Remove colunas desnecessárias e renomeia as colunas
    dfCidades = dfCidades.drop(['Regional', 'Unnamed: 3'], axis=1)
    dfCidades.columns = ['UF', 'MUN', 'CLASSE']
    
    # Cria uma coluna combinando UF e MUN para identificação única da cidade
    dfCidades['UF_MUN'] = dfCidades['UF'] + dfCidades['MUN']
    dfCidades = dfCidades[['UF_MUN', 'CLASSE']]
    
    # Filtra classes indesejadas e contabiliza as classes
    # dfCidades['CLASSE'].value_counts()
    # dfCidades = dfCidades.query('CLASSE != "ND"')
    # dfCidades = dfCidades.query('CLASSE != "<100k hab"')
    # dadosEcq = 'exportar_ECQ1.csv'
    # Lê o arquivo ECQ e realiza o tratamento das datas
    try:
        df = pd.read_csv(dadosEcq, sep=';', parse_dates=[_COLUNA_DATA_])
    except:
        df = pd.read_csv(dadosEcq, sep=',', parse_dates=[_COLUNA_DATA_])

    df['DIA'] = pd.to_datetime(df[_COLUNA_DATA_])
    # D_jan = np.datetime64(_D_JANELA_)
    # D_2 = np.datetime64(_D_2_)
    # df = df[(df[_COLUNA_DATA_] >= D_jan)]
    # df = df[(df[_COLUNA_DATA_] < D_2)]

    return df, dfCidades

def Tratamento(df, dfCidades):
    '''
    This function performs the treatment of ECQ data.

    Parameters:
        - df (DataFrame): DataFrame of ECQ data.
        - dfCidades (DataFrame): DataFrame of city classifications.

    Returns:
        - group_ecq (DataFrame): Treated DataFrame of ECQ data.
    '''
    
    # Creates a deep copy of the input DataFrame
    ecq = df.copy(deep=True)

    # Selects the relevant columns
    Colunas = ['ENDERECO_ID', 'LATITUDE', 'LONGITUDE', 'UF',
               'REGIONAL', 'ANF', 'MUNICIPIO',
               'PERDA_EXC_DOWN', 'PERDA_EXC_FIRSTBYTE',
               'PERDA_EXC_JITTER', 'PERDA_EXC_LAT', 'PERDA_EXC_PKTLOSS',
               'PERDA_EXC_PKTLOSS_LOST', 'PERDA_EXC_UP',
               'TESTES_ECQ', 'TESTES_ECQ_OK']

    ecq = ecq[Colunas]

    # Aggregates the data by selected columns
    group_ecq = ecq.groupby(['ENDERECO_ID', 'LATITUDE', 'LONGITUDE', 'UF',
                             'REGIONAL', 'ANF', 'MUNICIPIO']).aggregate(
        {
            'PERDA_EXC_DOWN': 'sum',
            'PERDA_EXC_FIRSTBYTE': 'sum',
            'PERDA_EXC_JITTER': 'sum',
            'PERDA_EXC_LAT': 'sum',
            'PERDA_EXC_PKTLOSS': 'sum',
            'PERDA_EXC_PKTLOSS_LOST': 'sum',
            'PERDA_EXC_UP': 'sum',
            'TESTES_ECQ': 'sum',
            'TESTES_ECQ_OK': 'sum'
        }).reset_index()

    # Creates new calculated columns
    group_ecq['TESTES_ECQ_NOK'] = group_ecq['TESTES_ECQ'] - group_ecq['TESTES_ECQ_OK']
    group_ecq['TESTES*TESTES_ECQ_NOK'] = group_ecq['TESTES_ECQ'] * group_ecq['TESTES_ECQ_NOK']

    # Calculates errors related to Transmission Rate (TX)
    group_ecq['ERROS_TX'] = group_ecq[
        ['PERDA_EXC_JITTER', 'PERDA_EXC_LAT', 'PERDA_EXC_PKTLOSS', 'PERDA_EXC_PKTLOSS_LOST']
    ].max(axis=1)

    group_ecq['ERROS_TX_TOTAL'] = group_ecq['TESTES_ECQ'] * group_ecq['ERROS_TX']

    # Calculates errors related to Access
    group_ecq['ERROS_ACESSO'] = group_ecq[
        ['PERDA_EXC_DOWN', 'PERDA_EXC_UP']
    ].max(axis=1) - group_ecq['ERROS_TX']

    # Ensures that there are no negative values for access errors
    group_ecq['ERROS_ACESSO'] = group_ecq['ERROS_ACESSO'].clip(0)

    group_ecq['ERROS_ACESSO_TOTAL'] = group_ecq['TESTES_ECQ'] * group_ecq['ERROS_ACESSO']

    # Calculates weights and weightings
    TESTES_TOTAL = group_ecq['TESTES_ECQ'].sum()
    TESTES_OK = group_ecq['TESTES_ECQ_OK'].sum()

    group_ecq['PESO_ECQ'] = (group_ecq['TESTES_ECQ'] - group_ecq['TESTES_ECQ_OK']) / (
            TESTES_TOTAL - TESTES_OK)  # weight in total ECQ failures
    group_ecq['PONDERADO'] = TESTES_OK / TESTES_TOTAL - (
            TESTES_OK - group_ecq['TESTES_ECQ']) / (TESTES_TOTAL - group_ecq['TESTES_ECQ'])

    # Creates a UF_MUN column for later joining with dfCidades
    group_ecq['UF_MUN'] = group_ecq['UF'] + group_ecq['MUNICIPIO']
    
    # Performs the join with dfCidades and removes the temporary UF_MUN column
    group_ecq = group_ecq.merge(dfCidades, on='UF_MUN', how='left')
    group_ecq.drop(['UF_MUN'], axis=1, inplace=True)

    return group_ecq

def Priorizar(group_ecq):
    '''
    This function prioritizes ECQ data groups based on clusters.

    Parameters:
        - group_ecq (DataFrame): DataFrame with treated ECQ data.

    Returns:
        - group_ecq (DataFrame): DataFrame with priorities assigned to clusters.
    '''
    
    # Initialization of the K-means algorithm with parameters
    # low, medium and high priorities. 3 clusters
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1000)

    # Iteration over unique keys defined in _CHAVE_KMEANS_
    for chave in group_ecq[_CHAVE_KMEANS_].unique():
        try:
            # Clustering for the Total aspect
            df_total = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)][['TESTES_ECQ_NOK', 'TESTES*TESTES_ECQ_NOK']]
            kmeans.fit(df_total)
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ECQ'] = kmeans.fit_predict(df_total)
            
            # Aggregation and mapping of priorities
            ecq_clusters = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)].groupby(['PRIORIDADE_ECQ']).agg({'PESO_ECQ': 'sum'}).sort_values('PESO_ECQ', ascending=False)
            priority_mapping = {ecq_clusters.index[i]: f'{["Low", "Medium", "High"][i]} Prio' for i in range(3)}
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ECQ'] = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ECQ'].map(priority_mapping)

            # Clustering for the Access aspect
            df_acesso = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)][['ERROS_ACESSO', 'ERROS_ACESSO_TOTAL']]
            kmeans.fit(df_acesso)
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ACESSO'] = kmeans.fit_predict(df_acesso)
            
            # Aggregation and mapping of priorities for Access
            acesso_clusters = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)].groupby(['PRIORIDADE_ACESSO']).agg({'PESO_ECQ': 'sum'}).sort_values('PESO_ECQ', ascending=False)
            priority_mapping = {acesso_clusters.index[i]: f'{["Low", "Medium", "High"][i]} Prio' for i in range(3)}
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ACESSO'] = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_ACESSO'].map(priority_mapping)

            # Clustering for the Transport aspect
            df_transporte = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)][['ERROS_TX', 'ERROS_TX_TOTAL']]
            kmeans.fit(df_transporte)
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_TX'] = kmeans.fit_predict(df_transporte)
            
            # Aggregation and mapping of priorities for Transport
            tx_clusters = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave)].groupby(['PRIORIDADE_TX']).agg({'PESO_ECQ': 'sum'}).sort_values('PESO_ECQ', ascending=False)
            priority_mapping = {tx_clusters.index[i]: f'{["Low", "Medium", "High"][i]} Prio' for i in range(3)}
            group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_TX'] = group_ecq.loc[(group_ecq[_CHAVE_KMEANS_] == chave), 'PRIORIDADE_TX'].map(priority_mapping)
        except:
            pass

    return group_ecq

def Clusterizar(df):
    '''
    This function performs clustering of the data.

    Parameters:
        - df (DataFrame): DataFrame of the treated data.

    Returns:
        - df (DataFrame): DataFrame with clustering information.
    '''

    def search(score):
        '''
        Auxiliary function to determine the 
        appropriate number of clusters based 
        on K-means scores
        ''' 
        for n in range(1,len(score)):
            if score[n-1] - score[n] > -0.01 :
                return n
        return n

    # Removing duplicates and NaN values
    df = df.drop_duplicates(subset=['ENDERECO_ID'])
    df = df.dropna(subset=['PRIORIDADE_ECQ', 'PRIORIDADE_ACESSO', 'PRIORIDADE_TX'])
    df['UF_MUN'] = df['UF'] + df['MUNICIPIO']

    # Filtering the data for different priorities
    dfECQ = df.query('PRIORIDADE_ECQ != "Low Prio"').copy()
    dfECQ.rename(columns={'TESTES_ECQ_NOK': 'prioKmeans'}, inplace=True)
    dfECQ = dfECQ[['UF_MUN', 'ENDERECO_ID', 'LATITUDE', 'LONGITUDE', 'prioKmeans']]

    dfAccess = df.query('PRIORIDADE_ACESSO != "Low Prio"').copy()
    dfAccess.rename(columns={'ERROS_ACESSO': 'prioKmeans'}, inplace=True)
    dfAccess = dfAccess[['UF_MUN', 'ENDERECO_ID', 'LATITUDE', 'LONGITUDE','prioKmeans']]

    dfTX = df.query('PRIORIDADE_TX != "Low Prio"').copy()
    dfTX.rename(columns={'ERROS_TX': 'prioKmeans'}, inplace=True)
    dfTX = dfTX[['UF_MUN', 'ENDERECO_ID', 'LATITUDE', 'LONGITUDE', 'prioKmeans']]

    DataFrames = [dfECQ, dfAccess, dfTX]
    
    K_clusters = range(1, __MAX_CLUSTERS__)
    # Performing clustering for each dataframe

    for dataframe in DataFrames:
        for cidade in dataframe['UF_MUN'].unique():
            try:
                dftemp = dataframe.loc[(dataframe['UF_MUN'] == cidade)][['LATITUDE', 'LONGITUDE', 'prioKmeans']]
                kmeans = [KMeans(n_clusters=i, n_init = 100, random_state=42) for i in K_clusters]
                
                auxiliar = dftemp[['LATITUDE','LONGITUDE']] 
                
                score = [kmeans[i].fit(auxiliar).score(auxiliar) for i in range(len(kmeans))]
                n_clusters = search(score)
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=42)
                kmeans.fit(dftemp[['LATITUDE', 'LONGITUDE']], sample_weight=dftemp['prioKmeans'])
                dataframe.loc[(dataframe['UF_MUN'] == cidade), 'CLUSTER'] = kmeans.fit_predict(dftemp[['LATITUDE', 'LONGITUDE']], sample_weight=dftemp['prioKmeans'])
            except:
                dataframe.loc[(dataframe['UF_MUN'] == cidade), 'CLUSTER'] = 0

    # Adding clustering information to the original dataframes
    try:
        dfECQ['CLUSTER_ECQ'] = _ANO_MES_ + "_" + dfECQ['CLUSTER'].astype(int).astype(str) + "_" + dfECQ['UF_MUN']
        dfECQ = dfECQ[['ENDERECO_ID', 'CLUSTER_ECQ']]

        dfAccess['CLUSTER_ACESSO'] =  _ANO_MES_ + "_" + dfAccess['CLUSTER'].astype(int).astype(str) + "_" + dfAccess['UF_MUN']
        dfAccess = dfAccess[['ENDERECO_ID', 'CLUSTER_ACESSO']]

        dfTX['CLUSTER_TX'] = _ANO_MES_ + "_" + dfTX['CLUSTER'].astype(int).astype(str) + "_" + dfTX['UF_MUN']
        dfTX = dfTX[['ENDERECO_ID', 'CLUSTER_TX']]
    except:
        pass

    # Combining the clustering dataframes with the original dataframe
    df = df.drop(['UF_MUN'], axis=1)
    df = df.merge(dfECQ, on='ENDERECO_ID', how='left')
    df = df.merge(dfAccess, on='ENDERECO_ID', how='left')
    df = df.merge(dfTX, on='ENDERECO_ID', how='left')
    df['MES'] = _ANO_MES_

    return df


def PrepararSaida(df, debugar=False):
    '''
    Prepara os dados de saída para serem salvos em um arquivo ou para depuração.

    Parâmetros:
        - df (DataFrame): DataFrame com as informações tratadas e clusterizadas.
        - debugar (bool): Indicador de depuração. Se True, retorna o DataFrame completo.

    Retorna:
        - df (DataFrame): DataFrame de saída preparado para ser salvo em um arquivo.
    '''

    if debugar:
        # Retorna o DataFrame completo se a depuração estiver ativada
        return df
    else:
        # Seleciona colunas relevantes para a saída
        colunas = ['ENDERECO_ID', 'PRIORIDADE_ECQ',
                   'PRIORIDADE_ACESSO', 'PRIORIDADE_TX',
                   'CLUSTER_ECQ', 'CLUSTER_ACESSO', 'CLUSTER_TX']
        df = df[colunas]
        return df


def main(debugar=False):
    '''
    Função principal para processar e preparar dados para saída.

    Parâmetros:
        - debugar (bool): Indicador de depuração. Se True, retornará o DataFrame completo.

    Retorna:
        - dfSaida (DataFrame): DataFrame preparado para saída ou para depuração.
    '''

    # Passo 1: Ler os dados de entrada

    dfEntrada, dfCidades = LerDados('tb_ecq_agreg_2023-08-30.csv', 
                                    'classificacao_cidades.xlsx')
    
    dfEntrada.shape
    dfEntrada['REGIONAL'].value_counts()
#filtros
# 26/07 TSL e TSP
    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-08-21')
    condicao2 = dfEntrada['REGIONAL'] == 'TSL'
    condicao3 = dfEntrada['REGIONAL'] == 'TSP'
    rowsToDrop = dfEntrada[(condicao1 & condicao2) | (condicao1 & condicao3)].index
    dfEntrada = dfEntrada.drop(rowsToDrop)
    dfEntrada['REGIONAL'].value_counts()
    dfEntrada.shape

# # 16/08 TNE
    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-08-16')
    condicao2 = dfEntrada['REGIONAL'] == 'TNE'
    rowsToDrop = dfEntrada[condicao1 & condicao2].index
    dfEntrada = dfEntrada.drop(rowsToDrop)
    dfEntrada['REGIONAL'].value_counts()
    dfEntrada.shape

# # 31/08 TCO e TNO
    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-08-31')
    condicao2 = dfEntrada['REGIONAL'] == 'TCO'
    condicao3 = dfEntrada['REGIONAL'] == 'TNO'
    rowsToDrop = dfEntrada[(condicao1 & condicao2) | (condicao1 & condicao3)].index
    dfEntrada = dfEntrada.drop(rowsToDrop)
    dfEntrada.shape
    dfEntrada['REGIONAL'].value_counts()

# # 05 a 07/09 TCO/TNO/TNE
    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-09-07')
    condicao2 = dfEntrada['REGIONAL'] == 'TCO'
    condicao3 = dfEntrada['REGIONAL'] == 'TNO'
    condicao4 = dfEntrada['REGIONAL'] == 'TNE'
    rowsToDrop = dfEntrada[(condicao1 & condicao2) | 
                           (condicao1 & condicao3) |
                           (condicao1 & condicao4)].index
    dfEntrada = dfEntrada.drop(rowsToDrop)

    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-09-08')
    rowsToDrop = dfEntrada[(condicao1 & condicao2) | 
                           (condicao1 & condicao3) |
                           (condicao1 & condicao4)].index
    dfEntrada = dfEntrada.drop(rowsToDrop)
    
    condicao1 = dfEntrada['DIA'] == pd.Timestamp('2023-09-09')
    rowsToDrop = dfEntrada[(condicao1 & condicao2) | 
                           (condicao1 & condicao3) |
                           (condicao1 & condicao4)].index
    dfEntrada = dfEntrada.drop(rowsToDrop)
    dfEntrada['REGIONAL'].value_counts()



    # dfEntrada, dfCidades = LerDados('ECQ_AGREG_ANF81.csv', 
    #                                 'classificacao_cidades.xlsx')
    
    # Passo 2: Realizar o tratamento nos dados
    dfTratado = Tratamento(dfEntrada, dfCidades)
    
    # Passo 3: Priorizar os clusters de dados
    dfPriorizacao = Priorizar(dfTratado)

    # Passo 4: Realizar a clusterização
    dfCluster = Clusterizar(dfPriorizacao)

    #dfCluster.to_excel('Saida.xlsx', index=False)
    # Passo 5: Preparar os dados de saída
    dfSaida = PrepararSaida(dfCluster, True)
    dfSaida.to_excel('REFERENCIA_AGOSTO23_REV5.xlsx', index=False)
    
    return dfSaida



