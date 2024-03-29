import pandas as pd
import numpy as np

def criar_dataset_sintetico_ma(data, date_col, target_col, num_pontos_out_of_sample):
    """
    Cria um conjunto de dados sintético baseado em uma série temporal.

    Parâmetros:
    - data (pd.DataFrame): O DataFrame original contendo a série temporal.
    - date_col (str): Nome da coluna que representa a data.
    - target_col (str): Nome da coluna que será usada como alvo para preenchimento.
    - num_pontos_out_of_sample (int): Número de pontos a serem gerados para o conjunto de dados sintético.

    Retorna:
    - pd.DataFrame: DataFrame atualizado com os valores estimados para os próximos pontos.

    Exemplo:
    ```python
    # Exemplo de uso
    novo_dataset = criar_dataset_sintetico(data_original, 'Data', 'Alvo', 10)
    ```

    """
    # Ordena o DataFrame pela coluna de data
    data = data.sort_values(by=date_col)

    # Colunas que não são a coluna de data
    colunas_nao_date = [coluna for coluna in data.columns if coluna not in [target_col, date_col]]

    # Criando os próximos meses baseados no último mês conhecido
    ultimo_mes_conhecido = data[date_col].max()
    proximos_meses = pd.date_range(start=ultimo_mes_conhecido + pd.DateOffset(months=1), periods=num_pontos_out_of_sample, freq='MS')

    # Preenchendo iterativamente os próximos valores estimados
    for proximo_mes in proximos_meses:
        # Dicionário para armazenar os valores
        novo_registro = {date_col: proximo_mes}

        # Preenchendo as demais colunas não relacionadas à data
        for coluna in colunas_nao_date:
            # Calculando a média móvel ponderada para a coluna atual
            ma_estimado = (data[coluna].iloc[-2:].values * np.arange(1, 3)).sum() / np.arange(1, 3).sum()

            # Adicionando o valor estimado ao dicionário
            novo_registro[coluna] = ma_estimado

        # Adicionando os próximos valores estimados ao DataFrame
        data = data.append(novo_registro, ignore_index=True)

    return data

def criar_dataset_sintetico_sarimax(data, date_col, target_col, num_pontos_out_of_sample):
    # Ordena o DataFrame pela coluna de data
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col)

    # Colunas que não são a coluna de data
    colunas_nao_date = [coluna for coluna in data.columns if coluna not in [target_col, date_col]]

    # Criando os próximos meses baseados no último mês conhecido
    ultimo_mes_conhecido = data[date_col].max()
    proximos_meses = pd.date_range(start=ultimo_mes_conhecido + pd.DateOffset(months=1), periods=num_pontos_out_of_sample, freq='MS')

    # DataFrame para armazenar todos os novos registros
    novos_registros_df = pd.DataFrame()

    # Preenchendo iterativamente os próximos valores estimados
    for coluna in tqdm(colunas_nao_date, desc="Processando colunas"):
        # Selecionando a coluna atual
        serie_temporal = data[[date_col, coluna]].set_index(date_col)

        # Busca de melhores parâmetros com Auto ARIMA
        modelo_arima = auto_arima(serie_temporal[coluna], d=1, start_p=1, start_q=1,
                                  max_p=5, max_q=5, m=6,  D=1, start_P=1,  
                                  start_Q=1, max_P=5, max_Q=5, information_criterion='aic', 
                                  seasonal=True, trace=False, error_action='ignore', stepwise=True)

        # melhores parametros
        best_order = modelo_arima.order
        best_seasonal_order = modelo_arima.seasonal_order

        # modelo SARIMAX com os melhores parâmetros
        modelo_sarimax = SARIMAX(serie_temporal[coluna], order=best_order, seasonal_order=best_seasonal_order)
        resultado_sarimax = model.fit()

        # out of sample forecast
        forecast = resultado_sarimax.get_forecast(steps=num_pontos_out_of_sample)
        previsoes = forecast.predicted_mean

        # Adicionando os novos registros ao DataFrame temporário
        novos_registros_df[coluna] = previsoes

    # Adicionando os novos registros ao DataFrame original
    novos_registros_df = novos_registros_df.reset_index().rename(columns={"index":date_col})
    data = pd.concat([data, novos_registros_df], axis=1)

    return data, novos_registros_df
