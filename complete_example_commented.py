import pandas as pd
import itertools
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
import logging
import numpy as np
import csv
__CORTE__ = 0.2  # 0.1 graus. equivale a 10km aprox.


def voronoi(df1, dfAmostras1):
    '''
    Entrada:
    endid: tabela com os endids da tim no formado endid ; lat ; lon
    amostras: tabela com as amostras do ECQ no formado chave ; lat ; lon
    False tem o efeito contrario
    Saida:
    dfAmostras: a tabela original com uma nova coluna informando o
    endid por voronoi
    dfVizinhos: uma Tabela listando os endids e seus vizinhos por voronoi
    '''
    # Read data from Excel file

    #df = pd.read_excel('floriano.xlsx')
    #dfAmostras = pd.read_excel('amostrasFAKE_floriano.xlsx')
    #df = pd.read_excel(endid)
    #dfAmostras = pd.read_excel(amostras)
    
    df = df1.copy(deep=True)
    dfAmostras = dfAmostras1.copy(deep=True)
    dfErros = pd.DataFrame(columns=['amostra', 'lat', 'lon'])
    dfTemp = pd.DataFrame(columns=['amostra', 'lat', 'lon'] )
    Errros = []
    
    # df = dfSpazio.copy(deep = True)
    # dfAmostras = dftemp.copy(deep = True)
    df.columns = ['endid', 'lat', 'lon']
    dfAmostras.columns = ['amostra', 'lat', 'lon']

    dictLatEndid = dict(zip(df['endid'], df['lat']))
    dictLonEndid = dict(zip(df['endid'], df['lon']))

    # Extract data points from the DataFrame
    pontos = df.iloc[:, 1:].values.tolist()
    PontosAmostras = dfAmostras.iloc[:, 1:].values.tolist()

    # Cria uma tabela no DF com o ID de cada região
    # df['Regioes'] = vor.point_region
    df['Regioes'] = df.index
    dicionario = dict(zip(df['Regioes'], df['endid']))

    # Achando as regioes de cada ponto
    voronoi_kdtree = cKDTree(pontos)
    _, test_point_regions = voronoi_kdtree.query(PontosAmostras)
    dfAmostras['endid'] = test_point_regions
    # trocando o nome da regiao pelo endid
    dfAmostras['endid'] = dfAmostras['endid'].map(dicionario)

    # achando os vizinhos
    # Compute Delaunay triangulation
    tri = Delaunay(pontos)
    # Build a dictionary of neighboring vertices for each vertex

    # SciPy Antigo
    # for p in tri.vertices:
    #     for i, j in itertools.combinations(p, 2):
    #         neiList[i].add(j)
    #         neiList[j].add(i)

    # SciPy Novo
    neiList = defaultdict(set)
    indptr, indices = tri.vertex_neighbor_vertices
    for k in range(len(pontos)):
        neiList[k] = set(indices[indptr[k]:indptr[k+1]])

    dfVizinhos = pd.DataFrame.from_dict(neiList, orient='index')
    dfVizinhos['endid'] = dfVizinhos.index
    # aplicar o mapeamento a todas as colunas
    dfVizinhos = dfVizinhos.applymap(dicionario.get)
    last_col_name = dfVizinhos.columns[-1]
    dfVizinhos = dfVizinhos[[last_col_name] + list(dfVizinhos.columns[:-1])]

    dfMelted = dfVizinhos.melt(id_vars=['endid'], value_name='vizinhos')
    dfMelted = dfMelted[['endid', 'vizinhos']]
    dfMelted.dropna(axis=0, subset=['vizinhos'], inplace=True)
    dfVizinhos = dfMelted

    # Adiciona a coordenada do endId no dfAmostra e corta por
    # uma aproximacao. 0.2 em grau equivalem a 25 km.

    dfAmostras['latEndid'] = dfAmostras['endid'].map(dictLatEndid)
    dfAmostras['lonEndid'] = dfAmostras['endid'].map(dictLonEndid)
    dfAmostras['difLat'] = abs(abs(dfAmostras['lat'])
                               - abs(dfAmostras['latEndid']))
    dfAmostras['difLon'] = abs(abs(dfAmostras['lon'])
                               - abs(dfAmostras['lonEndid']))
    
    dfTemp =  dfAmostras.loc[dfAmostras['difLat'] >= __CORTE__]
    dfErros = pd.concat([dfErros, dfTemp], ignore_index=True)
    dfTemp =  dfAmostras.loc[dfAmostras['difLon'] >= __CORTE__]
    dfErros = pd.concat([dfErros, dfTemp], ignore_index=True)
    
    dfAmostras = dfAmostras.loc[dfAmostras['difLat'] < __CORTE__]
    dfAmostras = dfAmostras.loc[dfAmostras['difLon'] < __CORTE__]
    dfAmostras = dfAmostras[['amostra', 'endid']]

    dfVizinhos['latEndid'] = dfVizinhos['endid'].map(dictLatEndid)
    dfVizinhos['lonEndid'] = dfVizinhos['endid'].map(dictLonEndid)
    dfVizinhos['latViz'] = dfVizinhos['vizinhos'].map(dictLatEndid)
    dfVizinhos['lonViz'] = dfVizinhos['vizinhos'].map(dictLonEndid)
    dfVizinhos['difLat'] = abs(abs(dfVizinhos['latEndid'])
                               - abs(dfVizinhos['latViz']))
    dfVizinhos['difLon'] = abs(abs(dfVizinhos['lonEndid'])
                               - abs(dfVizinhos['lonViz']))
    dfVizinhos = dfVizinhos.loc[dfVizinhos['difLat'] < __CORTE__]
    dfVizinhos = dfVizinhos.loc[dfVizinhos['difLon'] < __CORTE__]
    dfVizinhos = dfVizinhos[['endid', 'vizinhos']]

    return dfAmostras, dfVizinhos, dfErros


def LerDados(arquivo):

    df = pd.read_csv(arquivo, sep=',') #, nrows=1000)
    df.dropna(subset=['LATITUDE_TIM_CLUSTER', 'LONGITUDE_TIM_CLUSTER'], axis=0)
    df['LATITUDE_TIM_CLUSTER'] = df['LATITUDE_TIM_CLUSTER'].str.replace(',', '.').astype(float)
    df['LONGITUDE_TIM_CLUSTER'] = df['LONGITUDE_TIM_CLUSTER'].str.replace(',', '.').astype(float)
    df['amostra'] = df['DIA'].astype(str) +"-" + df.index.astype(str)

    return df


def LerSpazio(arquivo):
    '''
    Ler e ajustar a planilha do Spazio alem de eliminar END_IDs proximos
    '''


    def EndidsProximos(LAT,LON, df1, DISTANCIA=0.0003):
        '''
        Função para encontrar os sites proximos e marcar um deles para 
        delecao
        DISTANCIA - É EM GRAUS DECIMAIS. 0.0003° = 30 metros
        '''

        for index, row in df1.iterrows():
            if ((np.abs(LAT - row['lat']) <= DISTANCIA) | (np.abs(LON - row['lon']) <= DISTANCIA)):
                return 1
            else:
                return 0

    


    #arquivo = r'dados\tb_ecq_ngnis.dsv'
    df = pd.read_csv(r'dados\Spazio - 2023-07-31.csv', sep =';')
    df = pd.read_csv(arquivo, sep = ';')
    df = df[['EndereçoID',
              'Classificacao',
              'Latitude',
              'Longitude',
              'Status',
              'Situação']]
    df.sort_values(by=df.columns[0], ascending=False, inplace= True)
    df = df.query('Classificacao == "ACESSO"')
    df = df.query("Status == 'Aquisitado'")
    df = df.query("Situação != 'Site Removido'")
    df = df[['EndereçoID',
              'Latitude',
              'Longitude']]
    df.columns = [['endid', 'lat', 'lon']]
    df['lat'] = df['lat'].replace({',': '.'})
    df['lon'] = df['lon'].replace({',': '.'})
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, level=0)
    df = pd.DataFrame(columns = ['endid', 'lat', 'lon'], data= df.values)

    df['deletar'] = df.apply(lambda x: EndidsProximos(x['lat'], x['lon'], df, 0.0003), axis=1)
      

    
    return df


def CalcularDados(df):

    ### calculos que dependem dos testes ecq (teste_ecq != 0 or !nulo)
    # filtrar para calculo
    group_ecq = df.dropna(subset=['TESTES_ECQ'], axis=0)
    group_ecq = group_ecq[(group_ecq.TESTES_ECQ > 0)]
    # selecionar as colunas que agregam
    group_ecq  = group_ecq[['DIA',
                            'PERDA_EXC_DOWN (bps)',
                            'PERDA_EXC_FIRSTBYTE (bps)',
                            'PERDA_EXC_JITTER (bps)',
                            'PERDA_EXC_LAT (bps)',
                            'PERDA_EXC_PKTLOSS (bps)',
                            'PERDA_EXC_PKTLOSS_LOST (bps)',
                            'PERDA_EXC_UP (bps)',
                            'TESTES_CORE_OK',
                            'TESTES_ECQ',
                            'TESTES_ECQ_OK',
                            'AVGECQ_FIRSTBYTE',
                            'ECQ_DOWNLOAD (Mbps)',
                            'ECQ_JITTER',
                            'ECQ_LATENCIA',
                            'ECQ_PKTLOSS_DISCARD',
                            'ECQ_PKTLOSS_OLD (bps)',
                            'ECQ_UPLOAD (Mbps)',
                            'endid']]
    # agrega
    group_ecq = group_ecq.groupby(['DIA','endid']).aggregate({'PERDA_EXC_DOWN (bps)':'sum',
                                                         'PERDA_EXC_FIRSTBYTE (bps)':'sum',
                                                         'PERDA_EXC_JITTER (bps)':'sum',
                                                         'PERDA_EXC_LAT (bps)':'sum',
                                                         'PERDA_EXC_PKTLOSS (bps)':'sum',
                                                         'PERDA_EXC_PKTLOSS_LOST (bps)':'sum',
                                                         'PERDA_EXC_UP (bps)':'sum',
                                                         'TESTES_CORE_OK':'sum',
                                                         'TESTES_ECQ':'sum',
                                                         'TESTES_ECQ_OK':'sum',
                                                         'AVGECQ_FIRSTBYTE':'mean',
                                                         'ECQ_DOWNLOAD (Mbps)':'mean',
                                                         'ECQ_JITTER':'mean',
                                                         'ECQ_LATENCIA':'mean',
                                                         'ECQ_PKTLOSS_DISCARD':'mean',
                                                         'ECQ_PKTLOSS_OLD (bps)':'mean',
                                                         'ECQ_UPLOAD (Mbps)':'mean'})
    

    ### calculos reliability (reliability_total != 0 ou !nulo)
    # filtrar para calculo
    group_reliab = df.dropna(subset=['RELIABILITY_TOTAL'], axis=0)
    group_reliab = group_reliab[(group_reliab.RELIABILITY_TOTAL > 0)]
    # selecionar as colunas que agregam
    group_reliab  = group_reliab[['DIA',
                                  'RELIABILITY_TOTAL',
                                  'RELIABILITY_OK',
                                  'endid']]
    # agrega
    group_reliab = group_reliab.groupby(['DIA','endid']).aggregate({'RELIABILITY_TOTAL':'sum',
                                                                    'RELIABILITY_OK':'sum'})

    ### calcula novas colunas
    group_ecq['PERDA_EXC_DOWN_PERC'] = group_ecq['PERDA_EXC_DOWN (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_FIRSTBYTE_PERC'] = group_ecq['PERDA_EXC_FIRSTBYTE (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_JITTER_PERC'] = group_ecq['PERDA_EXC_JITTER (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_LAT_PERC'] = group_ecq['PERDA_EXC_LAT (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_PKTLOSS_PERC'] = group_ecq['PERDA_EXC_PKTLOSS (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_PKTLOSS_LOST_PERC'] = group_ecq['PERDA_EXC_PKTLOSS_LOST (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['PERDA_EXC_UP_PERC'] = group_ecq['PERDA_EXC_UP (bps)']/group_ecq['TESTES_ECQ']
    group_ecq['CCQ_KPI'] = group_ecq['TESTES_CORE_OK']/group_ecq['TESTES_ECQ']
    group_ecq['ECQ_KPI'] = group_ecq['TESTES_ECQ_OK']/group_ecq['TESTES_ECQ']
    group_reliab['RELIABILITY'] = group_reliab['RELIABILITY_OK']/group_reliab['RELIABILITY_TOTAL']
    
    # junta dfs
    grouped = pd.merge(group_ecq, group_reliab, on=['DIA','endid'], how='outer')

    # calculo das colunas que dependem do reliability
    grouped['CCQ'] = grouped['CCQ_KPI']*grouped['RELIABILITY']
    grouped['ECQ'] = grouped['ECQ_KPI']*grouped['RELIABILITY']
    grouped.reset_index(inplace=True)

    return grouped


def main():
    
    dfdados = LerDados(r'dados/TB_FT_ECQ_DAILY_FULL.csv')
    dfSpazio = LerSpazio(r'dados\Spazio - 2023-07-31.csv')
    #dfCgi = pd.read_excel("dados/cgi.xlsx")
    data = {'amostra': ['amostra1'], 'lat': [-3.73], 'lon': [-38.53]}
    dfAmostraAgeu = pd.DataFrame(data, columns = ['amostra', 'lat', 'lon'])
    # Exemplos
    dftemp = dfdados[['amostra','LATITUDE_TIM_CLUSTER','LONGITUDE_TIM_CLUSTER']]
    dftemp.columns = ['amostra', 'lat', 'lon']
    dfvoronoi, dfVizinhos, dfErros = voronoi(dfSpazio, dftemp)
    dfvoronoi =dfvoronoi[['amostra', 'endid']]
    dfdados = pd.merge(dfdados, dfvoronoi, on='amostra', how='left')
    dfdados.drop(['amostra'], axis=1, inplace=True)
    dfCalculado = CalcularDados(dfdados)
    dfvoronoi, dfVizinhos, dfErros = voronoi(dfSpazio, dfCalculado)


main()


##### teste ##### e ##### validação #####
# teste = grouped.groupby('DIA').aggregate({'PERDA_EXC_DOWN (bps)':'sum',
#                                             'PERDA_EXC_FIRSTBYTE (bps)':'sum',
#                                             'PERDA_EXC_JITTER (bps)':'sum',
#                                             'PERDA_EXC_LAT (bps)':'sum',
#                                             'PERDA_EXC_PKTLOSS (bps)':'sum',
#                                             'PERDA_EXC_PKTLOSS_LOST (bps)':'sum',
#                                             'PERDA_EXC_UP (bps)':'sum',
#                                             'TESTES_CORE_OK':'sum',
#                                             'TESTES_ECQ':'sum',
#                                             'TESTES_ECQ_OK':'sum',
#                                             'AVGECQ_FIRSTBYTE':'mean',/
#                                             'ECQ_DOWNLOAD (Mbps)':'mean',
#                                             'ECQ_JITTER':'mean',
#                                             'ECQ_LATENCIA':'mean',
#                                             'ECQ_PKTLOSS_DISCARD':'mean',
#                                             'ECQ_PKTLOSS_OLD (bps)':'mean',
#                                             'ECQ_UPLOAD (Mbps)':'mean',
#                                             'RELIABILITY_TOTAL':'sum',
#                                             'RELIABILITY_OK':'sum'})

# teste['PERDA_EXC_DOWN_PERC'] = teste['PERDA_EXC_DOWN (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_FIRSTBYTE_PERC'] = teste['PERDA_EXC_FIRSTBYTE (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_JITTER_PERC'] = teste['PERDA_EXC_JITTER (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_LAT_PERC'] = teste['PERDA_EXC_LAT (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_PKTLOSS_PERC'] = teste['PERDA_EXC_PKTLOSS (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_PKTLOSS_LOST_PERC'] = teste['PERDA_EXC_PKTLOSS_LOST (bps)']/teste['TESTES_ECQ']
# teste['PERDA_EXC_UP_PERC'] = teste['PERDA_EXC_UP (bps)']/teste['TESTES_ECQ']
# teste['CCQ_KPI'] = teste['TESTES_CORE_OK']/teste['TESTES_ECQ']
# teste['ECQ_KPI'] = teste['TESTES_ECQ_OK']/teste['TESTES_ECQ']
# teste['RELIABILITY'] = teste['RELIABILITY_OK']/teste['RELIABILITY_TOTAL']
# teste['CCQ'] = teste['CCQ_KPI']*teste['RELIABILITY']
# teste['ECQ'] = teste['ECQ_KPI']*teste['RELIABILITY']
# teste.to_excel('revisao_agreg.xlsx', index = False)
# grouped.to_excel('output_voronoi_ecq_agreg.xlsx', index = True)