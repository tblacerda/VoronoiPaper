import itertools
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

__CORTE__ = 0.1  # 0.1 degrees. equivalent to approximately 10km.

def main(endid, amostras):
    '''
    Input:
    endid: table with the endids of tim in the format endid ; lat ; lon
    amostras: table with the ECQ samples in the format key ; lat ; lon
    False has the opposite effect
    Output:
    dfAmostras: the original table with a new column informing the
    endid by voronoi
    dfVizinhos: a Table listing the endids and their neighbors by voronoi
    '''
    # Read data from Excel file
    df = pd.read_excel('floriano.xlsx')
    dfAmostras = pd.read_excel('amostrasFAKE_floriano.xlsx')

    df.columns = ['endid', 'lat', 'lon']
    dfAmostras.columns = ['amostra', 'lat', 'lon']

    dictLatEndid = dict(zip(df['endid'], df['lat']))
    dictLonEndid = dict(zip(df['endid'], df['lon']))

    # Extract data points from the DataFrame
    pontos = df.iloc[:, 1:].values.tolist()
    PontosAmostras = dfAmostras.iloc[:, 1:].values.tolist()

    # Create a table in DF with the ID of each region
    df['Regioes'] = df.index
    dicionario = dict(zip(df['Regioes'], df['endid']))

    # Finding the regions of each point
    voronoi_kdtree = cKDTree(pontos)
    _, test_point_regions = voronoi_kdtree.query(PontosAmostras)
    dfAmostras['endid'] = test_point_regions
    # changing the name of the region by endid
    dfAmostras['endid'] = dfAmostras['endid'].map(dicionario)

    # finding the neighbors
    # Compute Delaunay triangulation
    tri = Delaunay(pontos)
    # Build a dictionary of neighboring vertices for each vertex
    neiList = defaultdict(set)

    # SciPy New
    neiList = defaultdict(set)
    indptr, indices = tri.vertex_neighbor_vertices
    for k in range(len(pontos)):
        neiList[k] = set(indices[indptr[k]:indptr[k+1]])

    dfVizinhos = pd.DataFrame.from_dict(neiList, orient='index')
    dfVizinhos['endid'] = dfVizinhos.index
    # apply the mapping to all columns
    dfVizinhos = dfVizinhos.applymap(dicionario.get)
    last_col_name = dfVizinhos.columns[-1]
    dfVizinhos = dfVizinhos[[last_col_name] + list(dfVizinhos.columns[:-1])]

    dfMelted = dfVizinhos.melt(id_vars=['endid'], value_name='vizinhos')
    dfMelted = dfMelted[['endid', 'vizinhos']]
    dfMelted.dropna(axis=0, subset=['vizinhos'], inplace=True)
    dfVizinhos = dfMelted

    # Add the endId coordinate in dfAmostra and cut by
    # an approximation. 0.2 in degree are equivalent to 25 km.
    dfAmostras['latEndid'] = dfAmostras['endid'].map(dictLatEndid)
    dfAmostras['lonEndid'] = dfAmostras['endid'].map(dictLonEndid)
    dfAmostras['difLat'] = abs(abs(dfAmostras['lat'])
                               - abs(dfAmostras['latEndid']))
    dfAmostras['difLon'] = abs(abs(dfAmostras['lon'])
                               - abs(dfAmostras['lonEndid']))
    dfAmostras = dfAmostras.loc[dfAmostras['difLat'] < 0.2]
    dfAmostras = dfAmostras.loc[dfAmostras['difLon'] < 0.2]
    dfAmostras = dfAmostras[['amostra', 'lat', 'lon', 'endid']]

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

    return dfAmostras, dfVizinhos

# Examples
dfAmostras, dfVizinhos = main('floriano.xlsx',
                              'amostrasFAKE_floriano.xlsx')

dfAmostras, dfVizinhos = main('ESTACOES_recife.xlsx',
                              'amostrasFAKE_recife.xlsx')

dfAmostras, dfVizinhos = main('endids_tne.xlsx',
                              'amostrasECQ.xlsx')

#dfAmostras.to_clipboard(index=False)
dfVizinhos.to_clipboard(index=False)
