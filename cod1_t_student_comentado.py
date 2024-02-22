import pandas as pd
import datetime
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta, date
import statsmodels.api as sm
from statistics import stdev
from tqdm import tqdm
import sys
import warnings
import os.path
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# Define constants
__DIAS_MAX__ = 45    # Maximum population considered
__DIAS_MIN__ = 30    # Minimum population considered
__DIAS_AMOSTRA__ = 7 # At least __DIAS_AMOSTRA__ / 2.
                     # Sample of recent days to be evaluated
__CONFIANCA__ = 0.95 # Confidence interval 0.95
__MEDIA_MOVEL__ = 2  # Moving average days for the long window.

# Get current directory
CURR_DIR = os.getcwd()
print(CURR_DIR)

def JanelaCurta(corte, dias):
    ''' Returns the days of the short window in a list
        D_BD is the date to get in the Database. From this date
    '''
    Janela_curta = []
    for i, dia in enumerate(range(1, dias + 1, 1)):
        D = pd.to_datetime(corte + timedelta(days=dia))
        Janela_curta.append(D)

    return set(Janela_curta)

def testTstudent(mediaPopulacao, StdDevPop, df, confianca):
    '''
    Uses the same as the previous function, but uses the statistics
    package of python
    '''
    intervalo = stats.t.interval(
        alpha=confianca,
        df=df,
        loc=mediaPopulacao,
        scale=StdDevPop)

    return intervalo

def AnaliseEstatistica(df, janela_curta):
    '''
    Statistical analysis.
    [mediaPop, intervaloInf, intervaloSup, mediaAmostra,
    DesvioAmostra, Resultado]
    '''
    mediaAmostra = -1
    desvioPadraoPop = -1
    N = -1
    mediaPop = -1

    try:
        df = df.iloc[- __DIAS_MAX__:]
        df_amostra = df.copy(deep=True)
        df_amostra = df_amostra.query("Dia in @janela_curta")
        UltimaAmostra = df_amostra['value'].iloc[-1]
        df = df.iloc[:df.shape[0]-__DIAS_AMOSTRA__, :]
        df['MediaMovel'] = df['value'].rolling(__MEDIA_MOVEL__).mean()
        df.dropna(how='any', axis=0, inplace=True)
        mediaAmostra = df_amostra['value'].mean()
        desvioPadraoPop = stdev(df['MediaMovel'])
        N = df['MediaMovel'].count()
        mediaPop = df['MediaMovel'].mean()
        if len(janela_curta & set(df_amostra['Dia'])) >= (__DIAS_AMOSTRA__ / 2):
            intervalo = testTstudent(mediaPop, desvioPadraoPop, N-1, __CONFIANCA__)
            intervalo = list(intervalo)

            if df.shape[0] >= __DIAS_MIN__:
                resultado = 'Constante'
                if mediaAmostra < intervalo[0] and UltimaAmostra < intervalo[0]:
                    resultado = 'Diminuição'
                elif mediaAmostra > intervalo[1] and UltimaAmostra > intervalo[1]:
                    resultado = 'Aumento'


                new_row = [mediaPop, intervalo[0], intervalo[1], mediaAmostra, desvioPadraoPop, resultado]
                return new_row
            else:
                resultado = 'Sem dados suficientes'
                new_row = [mediaPop, -1, -1, mediaAmostra, desvioPadraoPop, resultado]
                return new_row
    except:
        resultado = 'Erro!'
        new_row = [-1, -1, -1, -1, -1, resultado]

    new_row = [mediaPop, -1, -1, mediaAmostra, desvioPadraoPop, resultado]
    return new_row

def Tratamento(df):
    '''
    Basic treatment of the input df
    return DF
    '''
    df.columns = ['Dia', 'value']
    df['Dia'] = pd.to_datetime(df['Dia'], format='%d/%m/%Y')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', subset=['value'], inplace=True)
    df.drop_duplicates(inplace=True, subset=['Dia', 'value'])
    df.sort_values('Dia', ascending=True, inplace=True)
    df['value'] = df['value'].round(3)

    return df

def main(Entrada, DataCorte):
    '''
    Entrada: Excel file with the ECQ input spreadsheet
    DataCorte: Date in yyyy-mm-dd format
    Saida: File that will be written on the output
    '''
    df_entrada = pd.read_excel(Entrada)
    DataCorte = datetime.strptime(DataCorte, "%Y-%m-%d")
    df_tratado = Tratamento(df_entrada)

    janela_curta = JanelaCurta(DataCorte, __DIAS_AMOSTRA__)

    if df_tratado.shape[0] >= __DIAS_MIN__:
            try:
                SAIDA = AnaliseEstatistica(df_tratado, janela_curta)
            except Exception:
                resultado = 'Sem dados suficientes'
                SAIDA = [-1, -1, -1, -1, -1, resultado]
    else:
        resultado = 'Sem dados suficientes'
        SAIDA = [-1, -1, -1, -1, -1, resultado]

    return SAIDA

main(Entrada, DataCorte)
