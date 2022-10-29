# Time-Series-to-COVID-19-in-USA
Projeto que contempla o Bootcamp em Data Science da DIO

Aplicação de séries temporais e modelo preditivo ARIMA para prever a disseminação do COVID-19 nos Estados Unidos através de um dataset.

Vale ressaltar que não foi plotado os últimos gráficos do projeto, portanto segue neste README.

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

![confirmeds](https://user-images.githubusercontent.com/63620777/198844074-2576f0c7-066a-4b70-8db3-a8b8a666f866.PNG)

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

![deaths](https://user-images.githubusercontent.com/63620777/198844092-cbe1cec4-44b2-457d-bdb2-45a96138655d.PNG)
