from datetime import timedelta, datetime # CALCULO DA DIFERENÇA DE DATAS
import streamlit as st # FRONT-END
import pandas as pd # TRATAMENTO DOS DADOS

@st.cache_data(ttl=timedelta(hours=1)) # Força a reexecução a cada 1 hora
def load_and_prepare_data(csv_path):

    # 1. DEFINIÇÃO DO INTERVALO DE TEMPO (Últimos 24 meses até ontem)

    # Data de Fim: Ontem
    data_fim_historico = datetime.now().date()
    qtde_year = 1
    # Data de Início: 24 meses antes de ontem (aproximadamente 730 dias)
    # Usaremos um a abordagem mais precisa para 24 meses:
    data_inicio_historico = data_fim_historico.replace(year=data_fim_historico.year - qtde_year)

#    st.write(f"Intervalo do Histórico: {data_inicio_historico.strftime('%Y-%m-%d')} a {data_fim_historico.strftime('%Y-%m-%d')}") # Opcional, para debug

    # Carrega base de eventos

    """Carrega os dados e filtra rotas com histórico suficiente."""
    # Carrega base de eventos
    try:
        df = pd.read_csv(csv_path, sep=";") # Uso em produção
        #df = func_atenuados()['post'] # Para ambiente interno
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame(), [], 0

    # Transforma a data de Abertura em aaaa/mm/dd e garante o tipo datetime
    df["Abertura_BA"] = pd.to_datetime(df["Abertura_BA"], errors='coerce')
    df.dropna(subset=['Abertura_BA'], inplace=True)

    # Garante que a coluna está "limpa" para o filtro, removendo a parte da hora
    df["Abertura_BA"] = df["Abertura_BA"].dt.floor("D")

    # 2. FILTRO TEMPORAL NO DATAFRAME BASE
    df = df[
        (df["Abertura_BA"].dt.date >= data_inicio_historico) &
        (df["Abertura_BA"].dt.date <= data_fim_historico)
    ].copy()

    # Transforma a data de Abertura em aaaa/mm/dd
    df["Abertura_BA"] = pd.to_datetime(df["Abertura_BA"], errors='coerce').dt.floor("D")
    df.dropna(subset=['Abertura_BA'], inplace=True)

    # Filtra rotas válidas (>70% meses com eventos)
    df['ano_mes'] = df['Abertura_BA'].dt.to_period('M')

    if df['ano_mes'].empty:
        return pd.DataFrame(), [], 0

    total_meses = (df['ano_mes'].max() - df['ano_mes'].min()).n + 1
    avaliador = 0.70
    rotas_eventos = df['Rota_Afetada'].value_counts()

    # Abaixo, usando o filtro original do usuário:
    rotas_avaliadas = rotas_eventos[rotas_eventos > (total_meses * avaliador)].index
    df = df[df['Rota_Afetada'].isin(rotas_avaliadas)].copy()

    # Inserindo a coluna ano
    df['ano'] = df["Abertura_BA"].dt.year

    rotas_disponiveis = df["Rota_Afetada"].unique().tolist()

    return df, rotas_disponiveis, total_meses
