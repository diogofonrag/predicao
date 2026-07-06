import numpy as np # CALCULOS NUMÉRICOS
from sklearn.metrics import precision_score, recall_score#, f1_score
def color_coluna(s):
    """Aplica cores de fundo baseadas no nível de risco."""
    colors = []
    for v in s:
        # Alto Risco (Prontidão Crítica)
        if v >= 0.85:
            colors.append('background-color: #ff5555') # Vermelho forte
        # Risco Moderado Alto (Inspeção Preditiva)
        elif v >= 0.70:
            colors.append('background-color: #ff9999') # Vermelho claro
        # Risco Moderado Baixo (Preventiva Focada)
        elif v >= 0.40:
            colors.append('background-color: #ffff99') # Amarelo claro
        # Baixo Risco
        else:
            colors.append('background-color: #99ff99') # Verde claro
    return colors

# --- FUNÇÃO DE DECAIMENTO TEMPORAL ---
def apply_decay(series, latest_date, lambda_decay):
    """Calcula a soma dos pesos exponenciais para uma série de datas de evento."""
    if series.empty:
        return 0

    # 1. Calcula a idade do evento em dias
    days_diff = (latest_date - series).dt.days
    # 2. Calcula o peso: e ^ (-lambda * dias)
    weights = np.exp(-lambda_decay * days_diff)
    # 3. Retorna a soma dos pesos (a contagem ponderada)
    return weights.sum()

def dentro_periodo(data, inicio, fim):
    """Verifica se uma data está dentro de um período sazonal, ignorando o ano."""
    # data é pd.Timestamp (do DataFrame), inicio/fim são datetime.date (do st.date_input)

    # 1. Normaliza o ano da data de referência (Timestamp) e converte para datetime.date.
    ref_data_normalized = data.replace(year=2000).date()

    # 2. Normaliza o ano das datas de início e fim
    ini_normalized = inicio.replace(year=2000)
    fim_normalized = fim.replace(year=2000)

    # 3. Compara objetos datetime.date com datetime.date
    # Caso de período que vira o ano (ex: Nov 15 a Jan 30)
    if ini_normalized <= fim_normalized:
        return int(ini_normalized <= ref_data_normalized <= fim_normalized)
    else:
        normalized_end_of_year = pd.Timestamp(year=2000, month=12, day=31).date()
        normalized_start_of_year = pd.Timestamp(year=2000, month=1, day=1).date()

        return int(
            (ini_normalized <= ref_data_normalized <= normalized_end_of_year) or
            (normalized_start_of_year <= ref_data_normalized <= fim_normalized)
        )

def optimize_threshold_for_f1(y_true, y_proba):
    """Encontra o limiar que maximiza o F1-Score."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Retorna o melhor limiar e as métricas resultantes com esse limiar
    y_pred_best = (y_proba >= best_threshold).astype(int)
    return best_threshold, precision_score(y_true, y_pred_best), recall_score(y_true, y_pred_best)
