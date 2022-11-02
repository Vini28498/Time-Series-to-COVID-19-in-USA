# Importando as bibliotecas
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Lendo arquivo
df = pd.read_csv('D:/Unknown folder/Machine Enginner/Covid DIO/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])
df.head()

# Metadados
df.dtypes

# Tratamento REGEX para colunas
import re
def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()
    
df.columns = [corrige_colunas(col) for col in df.columns]

# Visualizando a quantidade de registros para cada País
df.countryregion.value_counts()

# Filtrando os EUA como nosso alvo
us = df.loc[df.countryregion == 'US']

# Visualizando os dados de casos confirmados para EUA
us_confirmerd = df.loc[
    (df.countryregion == 'US') & 
    (df.confirmed > 0)
]
us_confirmerd

# Plotando gráfico de casos confirmados
px.line(us_confirmerd, 'observationdate', 'confirmed',
        title = 'Evolução da COVID-19 nos EUA'
)

# Função para calcular novos casos 
us_confirmerd['novos_casos'] = list(map(
    lambda x: 0 if (x==0) else us_confirmerd['confirmed'].iloc[x] - us_confirmerd['confirmed'].iloc[x-1],
    np.arange(us_confirmerd.shape[0])
))

us_confirmed_novos_casos = us_confirmerd.loc[us_confirmerd.novos_casos > 0]

us_confirmed_novos_casos

# Plotando novos casos por dia nos EUA
px.line(us_confirmed_novos_casos, 'observationdate', 'novos_casos',
        title = 'Novos casos por dia nos EUA'
)

# Plotando casos de morte nos EUA
fig = go.Figure()

fig.add_trace(
    go.Scatter(x = us_confirmed_novos_casos.observationdate, y = us_confirmed_novos_casos.deaths, name= 'Mortes',
    mode='lines+markers', line={'color':'red'})
)

fig.update_layout(title = 'Mortes por COVID-19 no EUA')

fig.show()

"""FORMULA: taxa_crescimento = (present/past)**(1-n)-1"""

# Função para calcular a taxa de crescimento médio por dia
def taxa_crescimento(data, variable, data_inicio = None, data_fim = None):
    # Definir a primeira data disponível
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    # Definir as variáveis
    past = data.loc[data.observationdate == data_inicio, variable].values[0]
    present = data.loc[data.observationdate == data_fim, variable].values[0]

    # Definir o número de pontos
    n = (data_fim - data_inicio).days

    # Calculo
    taxa = (present/past)**(1/n) - 1

    return taxa*100

# Taxa de cresimento médio de acordo com o período total
taxa_crescimento(us_confirmed_novos_casos,'confirmed')

# Função para calcular a taxa de crescimento médio por dia
def taxa_crescimento_diario(data, variable, data_inicio=None):
    # Definir a primeira data disponível
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    data_fim = data.observationdate.max()

    # Definir o número de pontos
    n = (data_fim - data_inicio).days
    
    # Calculo
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1, n+1)
    ))
    return np.array(taxas) * 100

# Taxa de cresimento médio por dia
tx_dia = abs(taxa_crescimento_diario(us_confirmed_novos_casos,'confirmed'))
tx_dia

# Plotando a taxa de crescimento por dia
day_1 = us_confirmed_novos_casos.observationdate.loc[us_confirmed_novos_casos.confirmed > 0].min()

px.line(x=pd.date_range(day_1, us_confirmed_novos_casos.observationdate.max())[1:],
        y= tx_dia, title= 'Taxa de Crescimento de casos confirmados nos EUA')

# importando bibliotecas para modelagm preditiva
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Criando série para casos confirmados
confirmados = us_confirmed_novos_casos.confirmed
confirmados.index = us_confirmed_novos_casos.observationdate
confirmados

# Decomposição dos casos confirmados
res = seasonal_decompose(confirmados, period=1)

# Plotando séries temporais para casos confirmados
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

# Criando série para casos de morte
mortes = us_confirmed_novos_casos.deaths
mortes.index = us_confirmed_novos_casos.observationdate
mortes

# Decomposição dos casos de morte
res1 = seasonal_decompose(mortes, period=1)

# Plotando séries temporais para casos de morte
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(mortes.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

# Importando biblioteca ARIMA
from pmdarima.arima import auto_arima

# Aplicando modelo de predição para casos confirmados
model_confirm = auto_arima(confirmados)

# Plotando predição para casos confirmados
fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y=model_confirm.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=model_confirm.predict(30), name='Forecast'
))

fig.update_layout(title='Previsão de casos confirmados nos EUA para os próximos 30 dias')
fig.show()

# Aplicando modelo de predição para casos de morte
model_death = auto_arima(mortes)

# Plotando predição para casos de morte
fig = go.Figure(go.Scatter(
    x=mortes.index, y=mortes, name='Observados'
))

fig.add_trace(go.Scatter(
    x=mortes.index, y=model_death.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=model_death.predict(30), name='Forecast'
))

fig.update_layout(title='Previsão de casos de morte nos EUA para os próximos 30 dias')
fig.show()