import streamlit as st
import pandas as pd
import numpy as np
import random
from func.functions import apply_decay, dentro_periodo, optimize_threshold_for_f1
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from datetime import timedelta, datetime

# CONSTANTES DE OTIMIZAÇÃO: Mantenha ou ajuste
TARGET_PRECISION = 0.75
LIMIAR_MINIMO_INICIAL = 0.10
INCREMENTO = 0.01
LIMIAR_MAXIMO = 1.00
LAMBDA_DECAY = 0.0250
IMPACTO_PER_OBRA_BOOST = 0.125
SEASONAL_IMPACT_BOOST = 0.05
HISTORICAL_OBRA_MAX_CONTRIBUTION = 0.0175
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Função Lagged Features (Corrigida para usar nomes de colunas limpos)
def calculate_lagged_features(df_group):

    # Colunas agora são limpas ('Abertura_BA' e 'Causa')
    COLUNA_DATA = "Abertura_BA"
    COLUNA_CAUSA = "Causa"

    # Garante que a coluna de data é do tipo datetime
    df_group[COLUNA_DATA] = pd.to_datetime(df_group[COLUNA_DATA])

    # 1. ORDENAÇÃO
    df_group = df_group.sort_values(COLUNA_DATA)

    # 2. Eventos de Falha (Causa != "nenhuma") nos últimos 30 dias
    df_group['eventos_ultimos_30d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            # A subtração é feita aqui: datetime - timedelta
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=30)) &
            (df_group[COLUNA_CAUSA] != "nenhuma")
        ).sum(),
        axis=1
    )

    # 3. Eventos de Obra de Terceiros nos últimos 60 dias
    df_group['obras_ultimos_60d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=60)) &
            (df_group[COLUNA_CAUSA] == "Obras de Terceiros")
        ).sum(),
        axis=1
    )

    # 4. Carga Alta nos últimos 30 dias
    df_group['carga_alta_ultimos_30d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=30)) &
            (df_group[COLUNA_CAUSA] == "Carga Alta")
        ).sum(),
        axis=1
    )

    # 5. Vandalismo nos últimos 30 dias
    df_group['vandalismo_ultimos_30d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=30)) &
            (df_group[COLUNA_CAUSA] == "Vandalismo")
        ).sum(),
        axis=1
    )

    # 6. Queda de Árvore nos últimos 30 dias
    df_group['arvore_ultimos_30d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=30)) &
            (df_group[COLUNA_CAUSA] == "Queda de Arvore")
        ).sum(),
        axis=1
    )

    # 7. Queimadas nos últimos 60 dias
    df_group['queimadas_ultimos_60d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=60)) &
            (df_group[COLUNA_CAUSA] == "Queimadas")
        ).sum(),
        axis=1
    )

    # 8. Atenuação nos últimos 30 dias
    df_group['atenuacao_ultimos_30d'] = df_group.apply(
        lambda row: (
            (df_group[COLUNA_DATA] < row[COLUNA_DATA]) &
            (df_group[COLUNA_DATA] >= row[COLUNA_DATA] - timedelta(days=30)) &
            (df_group[COLUNA_CAUSA] == "Atenuacao")
        ).sum(),
        axis=1
    )

    return df_group

def run_full_calculation(df_base, data_inicio_input, data_fim_input, obras_rotas_input, total_months):

    df = df_base.copy()
    latest_historical_date = df["Abertura_BA"].max()

    df["sazonalidade"] = df["Abertura_BA"].apply(
        lambda x: dentro_periodo(x, data_inicio_input, data_fim_input)
    )

    # ---- CÁLCULO DAS FEATURES PONDERADAS POR DECAIMENTO TEMPORAL ----
    with st.spinner("Calculando peso temporal dos eventos históricos (Longo Prazo)..."):

        weighted_features_list = []
        all_unique_rotas = df["Rota_Afetada"].unique()

        def calculate_and_store_decay(cause_name, column_name):

            # --- Lógica de Filtro ---
            if cause_name is None:
                # 1. Caso de 'total_eventos_ponderados' (filtra por 'nenhuma')
                df_filtered = df[df['Causa'] != "nenhuma"]
            else:
                # 2. Para causas específicas (filtra pela causa)
                df_filtered = df[df['Causa'] == cause_name]

            # 🚨 CORREÇÃO ESSENCIAL: Trata DataFrame filtrado vazio
            if df_filtered.empty:
                # Cria uma série de zeros com o índice de todas as rotas
                weighted_series = pd.Series(0.0, index=all_unique_rotas)
            else:
                # 1. Calcula o decaimento usando groupby e apply
                weighted_series = df_filtered.groupby("Rota_Afetada").apply(
                    lambda group: apply_decay(group["Abertura_BA"], latest_historical_date, LAMBDA_DECAY)
                )
                # Garante índice único (Correção de erro anterior)
                weighted_series = weighted_series.groupby(level=0).mean()

                # Reindexa para incluir rotas sem a causa (que ficariam com NaN) e preenche.
                weighted_series = weighted_series.reindex(all_unique_rotas).fillna(0)

            # 2. Define o nome da Series explicitamente
            weighted_series.name = column_name

            # 3. Garante que a Series é incluída na lista para concatenação
            weighted_features_list.append(weighted_series)

        # Executa o cálculo para todas as categorias
        calculate_and_store_decay(None, "total_eventos_ponderados")
        calculate_and_store_decay("Obras de Terceiros", "eventos_obra_ponderados")
        calculate_and_store_decay("Carga Alta", "eventos_carga_alta_ponderados")
        calculate_and_store_decay("Vandalismo", "eventos_vandalismo_ponderados")
        calculate_and_store_decay("Queda de Arvore", "eventos_arvore_ponderados")
        calculate_and_store_decay("Queimadas", "eventos_queimada_ponderados")
        calculate_and_store_decay("Atenuacao", "eventos_atenuacao_ponderados")

        expected_weighted_columns = [
            "total_eventos_ponderados", "eventos_obra_ponderados",
            "eventos_carga_alta_ponderados", "eventos_vandalismo_ponderados",
            "eventos_arvore_ponderados", "eventos_queimada_ponderados",
            "eventos_atenuacao_ponderados"
        ]

        # 1. Concatena as Series pelo eixo 1 (colunas). O índice é Rota_Afetada.
        # Esta linha agora não falhará.
        df_weighted_features_raw = pd.concat(weighted_features_list, axis=1)

        # 2. Reindexa o DataFrame para garantir a ordem das colunas e preenche zeros.
        # Usa dict.fromkeys para garantir que a lista de colunas não tenha duplicatas.
        unique_expected_columns = list(dict.fromkeys(expected_weighted_columns))

        # O reindex de colunas e fillna(0)
        df_weighted_features = df_weighted_features_raw.reindex(columns=unique_expected_columns).fillna(0)

    # ---- GERAÇÃO DO DATASET COMPLETO (LINHAS NEGATIVAS) ----
    with st.spinner("Gerando dados sintéticos para treinamento..."):
        rows = []
        for rota, grupo in df.groupby("Rota_Afetada"):

            # Verifica se grupo["ano_mes"] é uma série válida para min/max
            if grupo["ano_mes"].empty:
                continue

            if grupo["ano_mes"].min() > grupo["ano_mes"].max():
                all_months = [grupo["ano_mes"].iloc[0]]
            else:
                all_months = pd.period_range(grupo["ano_mes"].min(), grupo["ano_mes"].max(), freq="M")

            for mes in all_months:
                eventos_mes = grupo[grupo["ano_mes"] == mes]
                num_eventos_reais = len(eventos_mes)
                num_negative_samples = 0

                if num_eventos_reais > 0:
                    rows.append(eventos_mes)

                if num_eventos_reais == 1:
                    num_negative_samples = 1
                elif num_eventos_reais == 0:
                    num_negative_samples = 2

                if num_negative_samples > 0:
                    dias_com_evento = eventos_mes['Abertura_BA'].dt.day.tolist()
                    all_days_in_month = list(range(1, mes.days_in_month + 1))
                    dias_disponiveis = [d for d in all_days_in_month if d not in dias_com_evento]

                    if not dias_disponiveis:
                        dias_disponiveis = all_days_in_month

                    if len(dias_disponiveis) < num_negative_samples:
                        sampled_days = np.random.choice(dias_disponiveis, size=num_negative_samples, replace=True)
                    else:
                        sampled_days = np.random.choice(dias_disponiveis, size=num_negative_samples, replace=False)

                    datas = [pd.Timestamp(year=mes.year, month=mes.month, day=int(d)) for d in sampled_days]

                    temp = pd.DataFrame({
                        "Rota_Afetada": [rota] * num_negative_samples,
                        "ano_mes": [mes] * num_negative_samples,
                        "ano": [mes.year] * num_negative_samples,
                        "Causa": ["nenhuma"] * num_negative_samples,
                        "Abertura_BA": datas,
                        "sazonalidade": [dentro_periodo(d, data_inicio_input, data_fim_input) for d in datas],
                        "UF": [grupo["UF"].iloc[0]] * num_negative_samples
                    })
                    rows.append(temp)

        df_completo = pd.concat(rows, ignore_index=True)
        # 🚨 VERIFICAÇÃO DE SEGURANÇA CONTRA DATAFRAME VAZIO 🚨
        if df_completo.empty:
            st.error("O conjunto de dados histórico completo (base + sintético) está vazio. Não é possível continuar a previsão.")
            st.session_state.feature_importance = pd.DataFrame()
            st.session_state.ranking_rotas = pd.DataFrame()
            return pd.DataFrame() # Retorna de forma segura
        # ----------------------------------------------------
        df_completo["Abertura_BA"] = pd.to_datetime(df_completo["Abertura_BA"])

    # ---- ENGENHARIA DE FEATURES FINAIS ----
    df_completo = df_completo.sort_values(["Abertura_BA"])
    df_completo['obras'] = 0
    df_completo['falha_futuro'] = 1

    df_completo.loc[df_completo['Causa'] != "nenhuma", 'falha_futuro'] = 1
    df_completo.loc[df_completo['Causa'] == "nenhuma", 'falha_futuro'] = 0
    df_completo.loc[df_completo['Causa'] == "Obras de Terceiros", 'obras'] = 1

    # Garante que a coluna de merge (Rota_Afetada) seja string/object
    df_completo['Rota_Afetada'] = df_completo['Rota_Afetada'].astype(str)

    # 🚨 SUBSTITUIÇÃO DO MERGE POR MAP - EVITA SUFIXOS (_x, _y) 🚨
    # Adiciona as features Ponderadas no dataset completo usando map
    for col in df_weighted_features.columns:
        # df_weighted_features está indexado por Rota_Afetada e atua como um dicionário
        df_completo[col] = df_completo["Rota_Afetada"].map(df_weighted_features[col]).fillna(0)
    # -------------------------------------------------------------

    # --- Cálculo das Lagged Features (Eventos nos últimos N dias) ---
    with st.spinner("Calculando features de curto prazo (últimos 30/60 dias)..."):
        # Aplica o cálculo por Rota (agora usando a função corrigida, sem sufixos)
        df_completo = df_completo.groupby("Rota_Afetada", group_keys=False).apply(calculate_lagged_features)

    # Codificação (Usando nomes de coluna limpos)
    cod_rota = LabelEncoder()
    cod_causa = LabelEncoder()
    cod_uf = LabelEncoder()

    df_completo['Rota_Afetada_Encoded'] = cod_rota.fit_transform(df_completo['Rota_Afetada'])
    # Assumo que 'UF' não foi renomeado, o que é o caso se 'df_base' for limpo e não houver merge intermediário.
    df_completo['Causa_Encoded'] = cod_causa.fit_transform(df_completo['Causa'])
    df_completo['UF_Encoded'] = cod_uf.fit_transform(df_completo['UF'])

    # ---- TREINAMENTO DO MODELO (O restante do código de treinamento e previsão futura está OK) ----
    with st.spinner("Treinando modelo XGBoost..."):
        X_full = df_completo[[
            "obras", "Rota_Afetada_Encoded", "ano",
            # PONDERADAS
            "eventos_obra_ponderados", "total_eventos_ponderados",
            "eventos_carga_alta_ponderados", "eventos_vandalismo_ponderados",
            "eventos_arvore_ponderados", "eventos_queimada_ponderados",
            "eventos_atenuacao_ponderados",
            "UF_Encoded",
            # LAGGED
            "eventos_ultimos_30d", "obras_ultimos_60d",
            "carga_alta_ultimos_30d", "vandalismo_ultimos_30d",
            "arvore_ultimos_30d", "queimadas_ultimos_60d",
            "atenuacao_ultimos_30d"
        ]]
        y_full = df_completo["falha_futuro"]
        # ... (seu código de split, treinamento, calibração e métricas) ...

        split_date = df_completo["Abertura_BA"].quantile(0.75)
        train_mask = df_completo["Abertura_BA"] <= split_date
        test_mask = df_completo["Abertura_BA"] > split_date
        X_train = X_full[train_mask]
        y_train = y_full[train_mask]
        X_test_full = X_full[test_mask]
        y_test_full = y_full[test_mask]

        if X_test_full.empty:
            st.error("Dados de teste vazios. Ajuste o histórico ou o percentual de corte.")
            return

        # ---- AJUSTA O PESO CONFORME O MODELO ----
        contagem = df_completo['falha_futuro'].value_counts()
        negativos = contagem.get(0, 0)
        positivos = contagem.get(1, 0)
        peso_class_pos = negativos/positivos
        st.write(f"Valor do peso atribuido =\n{peso_class_pos}")
        st.write(f"Quantidade de positivos  =\n{positivos}")
        st.write(f"Quantidade de negativos  =\n{negativos}")

        modelo_xgb_base = xgb.XGBClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.02,
            scale_pos_weight=peso_class_pos,
            random_state=42, use_label_encoder=False, eval_metric='logloss', gamma=0.5, reg_lambda=0.1,
            early_stopping_rounds=50,
        )
        #modelo_cal.fit(X_train, y_train)

        # 1. Definir o tamanho da divisão para validação (ex: 10% do conjunto de treino)
        validation_size = 0.22

        # 2. Calcular o índice de corte
        # Como é uma série temporal, garantimos que a ordem não seja alterada (shuffle=False)
        # e que a divisão seja feita do início ao fim (o último 10% é o mais recente).
        corte = int(len(X_train) * (1 - validation_size))

        # 3. Separar os dados:

        # TREINO FINAL (Usado para o treinamento real do XGBoost)
        X_treino_final = X_train.iloc[:corte]
        y_treino_final = y_train.iloc[:corte]

        # VALIDAÇÃO (Usado APENAS para o Early Stopping)
        X_val = X_train.iloc[corte:]
        y_val = y_train.iloc[corte:]

        modelo_xgb_base.fit(
            X_treino_final, y_treino_final,
            eval_set=[(X_val, y_val)],  # Conjunto de validação
            verbose=False
        )

        best_n_estimators = modelo_xgb_base.best_iteration

        ############## Modelo final para não gerar overload ######################
        modelo_xgb_final = xgb.XGBClassifier(
            n_estimators=best_n_estimators, max_depth=7, learning_rate=0.05,
            scale_pos_weight=peso_class_pos,
            random_state=42, use_label_encoder=False, eval_metric='logloss', gamma=0.5, reg_lambda=0.1
        )

        tscv = TimeSeriesSplit(n_splits=3)
#        modelo_cal = CalibratedClassifierCV(modelo_xgb_final, cv=tscv, method='isotonic')
        modelo_cal = CalibratedClassifierCV(modelo_xgb_final, cv=tscv, method='sigmoid')
        modelo_cal.fit(X_treino_final, y_treino_final)

    # ---- CÁLCULO DAS MÉTRICAS DO MODELO ----
    X_test = X_test_full
    y_test = y_test_full
    st.write(f"Número de arvores = {best_n_estimators}")

    if not X_test.empty:
        y_proba = modelo_cal.predict_proba(X_test)[:, 1]
        y_pred = modelo_cal.predict(X_test)

        st.session_state.accuracy = accuracy_score(y_test, y_pred) * 100
        st.session_state.pr_auc = average_precision_score(y_test, y_proba)
        st.session_state.roc_auc = roc_auc_score(y_test, y_proba)

        melhor_limiar = None
        melhor_precisao = 0.0
        limiar_atual = LIMIAR_MINIMO_INICIAL

        LIMIAR_MINIMO_ROC_AUC = 0.70 # 70%

        if st.session_state.roc_auc > LIMIAR_MINIMO_ROC_AUC:
            st.success(f"✅ Modelo Aprovado! ROC-AUC: {st.session_state.roc_auc:.2%} (Acima de {LIMIAR_MINIMO_ROC_AUC:.0%}).")
            limiar_otimizado = 0.70

        else:
            st.warning(f"⚠️ ROC-AUC ({st.session_state.roc_auc:.2%}) abaixo de {LIMIAR_MINIMO_ROC_AUC:.0%}. Tentando otimizar o Limiar Tático...")
            try:
                limiar_otimizado, precision_otim, recall_otim = optimize_threshold_for_f1(y_test, y_proba)
                st.session_state.limiar_tatico_otimizado = limiar_otimizado
                st.session_state.precision_70 = precision_otim * 100
                st.session_state.recall_70 = recall_otim * 100
                st.info(f"✨ Otimização Concluída: Novo Limiar Tático: {limiar_otimizado:.2%}. (Precisão: {precision_otim:.2%}, Recall: {recall_otim:.2%})")

            except Exception as e:
                st.error(f"❌ Falha ao tentar otimizar o limiar. O ranking não será gerado. Erro: {e}")
                st.session_state.ranking_rotas = None
                st.session_state.feature_importance = None
                st.session_state.limiar_tatico_otimizado = None
                return # Interrompe a execução aqui se a otimização falhar

        while limiar_atual <= LIMIAR_MAXIMO:
            y_pred_otimizado = (y_proba >= limiar_atual).astype(int)
            precision = precision_score(y_test, y_pred_otimizado, zero_division=0)

            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_otimizado, labels=[0, 1]).ravel()
                recall = recall_score(y_test, y_pred_otimizado, zero_division=0)
            except ValueError:
                recall = 0.0
                fp = 0
                fn = (y_test == 1).sum()

            if precision >= TARGET_PRECISION or limiar_atual == LIMIAR_MAXIMO:
                st.session_state.precision_70 = precision
                st.session_state.recall_70 = recall
                st.session_state.falso_positivo_70 = fp
                st.session_state.falso_negativo_70 = fn
                st.session_state.limiar_tatico_otimizado = limiar_atual
                if precision >= TARGET_PRECISION:
                    break

            elif precision > melhor_precisao:
                melhor_precisao = precision
                st.session_state.limiar_tatico_otimizado_fallback = limiar_atual

            limiar_atual += INCREMENTO

        if st.session_state.precision_70 < TARGET_PRECISION and 'limiar_tatico_otimizado_fallback' in st.session_state:
            limiar_fallback = st.session_state.limiar_tatico_otimizado_fallback
            y_pred_otimizado = (y_proba >= limiar_fallback).astype(int)
            st.session_state.precision_70 = precision_score(y_test, y_pred_otimizado, zero_division=0)
            st.session_state.recall_70 = recall_score(y_test, y_pred_otimizado, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_otimizado, labels=[0, 1]).ravel()
            st.session_state.falso_positivo_70 = fp
            st.session_state.falso_negativo_70 = fn
            st.session_state.limiar_tatico_otimizado = limiar_fallback

        if 'limiar_tatico_otimizado_fallback' in st.session_state:
            del st.session_state.limiar_tatico_otimizado_fallback

    else:
        st.session_state.accuracy = 0.0
        st.session_state.pr_auc = 0.0


    # ---- CÁLCULO DA IMPORTÂNCIA DE FEATURE ----
    with st.spinner("Calculando a importância das features..."):
        r = permutation_importance(
            modelo_cal, X_test_full, y_test_full,
            n_repeats=500,
            random_state=42,
            scoring='average_precision'
        )

        importance_df = pd.DataFrame({
            'feature': X_full.columns,
            'importance_mean': r.importances_mean,
            'importance_std': r.importances_std
        }).sort_values(by='importance_mean', ascending=False).reset_index(drop=True)

        st.session_state.feature_importance = importance_df


    # ---- GERAÇÃO DA PREVISÃO PARA OS PRÓXIMOS 7 DIAS ----

    # 1. Define o período futuro (7 dias)
    future_dates = pd.date_range(latest_historical_date + pd.Timedelta(days=1), periods=7, freq='D')

    # 2. Prepara o DataFrame de predição futura
    future_rows = []
    # Pega TODAS as features de rota que são estáticas (Encoded, PONDERADAS e UF_Encoded)
    unique_routes_data = df_completo[[
        "UF", "UF_Encoded", "Rota_Afetada", "Rota_Afetada_Encoded",
        "eventos_obra_ponderados", "total_eventos_ponderados",
        "eventos_carga_alta_ponderados", "eventos_vandalismo_ponderados",
        "eventos_arvore_ponderados", "eventos_queimada_ponderados",
        "eventos_atenuacao_ponderados"
    ]].drop_duplicates("Rota_Afetada").copy()

    # 2.1. Funções Auxiliares para cálculo de Lagged (Short-Term Features)

    def calculate_lagged_count(causa, window_days, column_name):
        # Filtra o df_completo pelo período e causa, e conta por rota
        return df_completo[
            (df_completo['Abertura_BA'] > latest_historical_date - timedelta(days=window_days)) &
            (df_completo['Abertura_BA'] <= latest_historical_date) &
            (df_completo['Causa'] == causa)
        ].groupby("Rota_Afetada").size().reindex(unique_routes_data["Rota_Afetada"]).fillna(0).rename(column_name)

    # 2.2. Cálculo de TODAS as features Lagged de curto prazo (Último estado)

    # Total de Falhas (Causa != nenhuma)
    count_30d = df_completo[
        (df_completo['Abertura_BA'] > latest_historical_date - timedelta(days=30)) &
        (df_completo['Abertura_BA'] <= latest_historical_date) &
        (df_completo['Causa'] != "nenhuma")
    ].groupby("Rota_Afetada").size().reindex(unique_routes_data["Rota_Afetada"]).fillna(0).rename("eventos_ultimos_30d")

    # Obra de Terceiros
    count_60d_obra = calculate_lagged_count("Obras de Terceiros", 60, "obras_ultimos_60d")

    # NOVAS FEATURES LAGGED (Carga Alta, Vandalismo, Árvore, Queimadas, Atenuação)
    count_30d_carga = calculate_lagged_count("Carga Alta", 30, "carga_alta_ultimos_30d")
    count_30d_vandalismo = calculate_lagged_count("Vandalismo", 30, "vandalismo_ultimos_30d")
    count_30d_arvore = calculate_lagged_count("Queda de Arvore", 30, "arvore_ultimos_30d")
    count_60d_queimadas = calculate_lagged_count("Queimadas", 60, "queimadas_ultimos_60d")
    count_30d_atenuacao = calculate_lagged_count("Atenuacao", 30, "atenuacao_ultimos_30d")

    # 2.3. Adiciona TODAS as contagens de curto prazo (último estado)
    unique_routes_data = unique_routes_data.merge(count_30d.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_60d_obra.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_30d_carga.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_30d_vandalismo.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_30d_arvore.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_60d_queimadas.to_frame(), on="Rota_Afetada", how="left").fillna(0)
    unique_routes_data = unique_routes_data.merge(count_30d_atenuacao.to_frame(), on="Rota_Afetada", how="left").fillna(0)

    # 2.4. Cria as linhas futuras (agora com todas as 18 features)
    for date in future_dates:
        temp_df = unique_routes_data.copy()
        temp_df['Abertura_BA'] = date
        temp_df['ano'] = date.year
        # O LabelEncoder da Causa deve ter sido treinado com "nenhuma" (linha 313)
        temp_df['Causa_Encoded'] = cod_causa.transform(["nenhuma"])[0]
        temp_df['obras'] = 0.0
        temp_df["sazonalidade"] = dentro_periodo(date, data_inicio_input, data_fim_input)
        future_rows.append(temp_df)

    df_future = pd.concat(future_rows, ignore_index=True)

    # 3. Features para previsão (DEVE TER AS MESMAS 18 COLUNAS DE X_FULL)
    X_future = df_future[[
        "obras", "Rota_Afetada_Encoded", "ano",
        "eventos_obra_ponderados", "total_eventos_ponderados",
        "eventos_carga_alta_ponderados", "eventos_vandalismo_ponderados",
        "eventos_arvore_ponderados", "eventos_queimada_ponderados",
        "eventos_atenuacao_ponderados",
        "UF_Encoded",
        "eventos_ultimos_30d", "obras_ultimos_60d",
        "carga_alta_ultimos_30d", "vandalismo_ultimos_30d",
        "arvore_ultimos_30d", "queimadas_ultimos_60d",
        "atenuacao_ultimos_30d"
    ]]

    # 4. Previsão da probabilidade base
    df_future["prob_base"] = modelo_cal.predict_proba(X_future)[:, 1]

    # ---- APLICAÇÃO DOS IMPACTOS NO FUTURO (Post-Processing) ----
    df_future["multiplicador_risco"] = 1.0

    df_future["impacto_obra_hist_ratio"] = df_future["eventos_obra_ponderados"] / df_future["total_eventos_ponderados"].replace(0, 1)
    df_future["multiplicador_risco"] += df_future["impacto_obra_hist_ratio"] * HISTORICAL_OBRA_MAX_CONTRIBUTION
    df_future.loc[df_future["Rota_Afetada"].isin(obras_rotas_input), "multiplicador_risco"] += IMPACTO_PER_OBRA_BOOST
    df_future["multiplicador_risco"] += np.where(df_future["sazonalidade"] == 1, SEASONAL_IMPACT_BOOST, 0.0)

    df_future["prob_risco"] = df_future["prob_base"] * df_future["multiplicador_risco"]
    df_future["prob_risco"] = df_future["prob_risco"].clip(0, 1)

    # ---- GERAÇÃO DO RANKING ----
    ranking_rotas = df_future.groupby(["UF", "Rota_Afetada"])["prob_risco"].mean().reset_index()

    total_eventos_brutos = df_base[df_base['Causa'] != "nenhuma"].groupby("Rota_Afetada").size()
    avg_monthly_frequency = total_eventos_brutos / total_months
    ranking_rotas["Frequencia_Mensal_Media"] = ranking_rotas["Rota_Afetada"].map(avg_monthly_frequency).fillna(0)
    ranking_rotas["Rota_Em_Obra_Selecionada"] = np.where(ranking_rotas["Rota_Afetada"].isin(obras_rotas_input), "Sim", "Não")

    # Armazena o ranking no estado da sessão
    st.session_state.ranking_rotas = ranking_rotas

    # Retorna o DataFrame que contém as 18 features (útil para debug e análise)
    return df_completo
