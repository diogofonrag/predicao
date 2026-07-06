import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import datetime, timedelta

# ASSUMA que as importações abaixo estão corretas
from func.functions import color_coluna
from func.load_post import load_and_prepare_data
from func.func_prediction import run_full_calculation

# CONSTANTES
TABLE_PRED = 'feedback_tracker' 
limiar_roac_alvo = 0.7 # O objetivo de ROC-AUC

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Ranking de Risco - Telecom", layout="wide")

# =================================================================
# 1. CARREGAMENTO E INPUTS
# =================================================================
df_base, rotas_disponiveis, total_meses = load_and_prepare_data(r"data\Post.csv")

if df_base.empty:
    st.info("O DataFrame de eventos está vazio ou não pôde ser carregado.")
    st.stop()

st.title("Predição de Risco em Rotas de Fibra Óptica - PRRF")
st.markdown("---")

# Datas para o histórico
data_fim_historico = datetime.now().date()
qtde_year = 1
data_inicio_historico = data_fim_historico.replace(year=data_fim_historico.year - qtde_year)

st.subheader(f" 📅 Intervalo do Histórico: {data_inicio_historico} a {data_fim_historico}")

# Inputs de Regras de Negócio
col1, col2 = st.columns(2)
with col1:
    data_inicio = st.date_input("Início Período Sazonal", value=pd.to_datetime("2025-08-01").date())
with col2:
    data_fim = st.date_input("Fim Período Sazonal", value=pd.to_datetime("2025-10-15").date())

obras_rotas = st.multiselect("Rotas com obras ativas (Boost 12.5%)", rotas_disponiveis)

st.markdown("---")

# =================================================================
# 2. PROCESSO DE CÁLCULO (DISPARADO PELO BOTÃO)
# =================================================================
if st.button("Calcular Risco e Gerar Ranking", type="primary"):
    # Reseta estados para garantir um novo cálculo limpo
    st.session_state.ranking_rotas = None
    st.session_state.roc_auc = 0
    
    # Placeholder para feedback visual
    progresso_bar = st.progress(0)
    status_text = st.empty()
    
    max_tentativas = 100
    tentativas = 0
    
    # Loop de Otimização: Só para quando atingir o limiar ou o máximo de tentativas
    while st.session_state.get('roc_auc', 0) < limiar_roac_alvo and tentativas < max_tentativas:
        tentativas += 1
        
        # Atualiza interface
        porcentagem = tentativas / max_tentativas
        progresso_bar.progress(porcentagem)
        status_text.info(f"Otimizando modelo... Tentativa {tentativas}/{max_tentativas} (ROC-AUC atual: {st.session_state.get('roc_auc', 0):.4f})")
        
        # Executa o cálculo pesado
        run_full_calculation(df_base, data_inicio, data_fim, obras_rotas, total_meses)
    
    # Finaliza indicadores
    progresso_bar.empty()
    status_text.success(f"Cálculo finalizado em {tentativas} tentativas! ROC-AUC: {st.session_state.roc_auc:.4f}")

# =================================================================
# 3. EXIBIÇÃO E SALVAMENTO (SÓ APARECE SE HOUVER RESULTADO)
# =================================================================
if 'ranking_rotas' in st.session_state and st.session_state.ranking_rotas is not None:
    ranking_rotas = st.session_state.ranking_rotas.copy()
    limiar_otimizado = st.session_state.get('limiar_tatico_otimizado', 0.45)

    # Sidebar com métricas
    st.sidebar.subheader("Métricas de Performance")
    st.sidebar.metric("ROC-AUC", f"{st.session_state.roc_auc:.4f}")
    st.sidebar.metric("Precisão", f"{st.session_state.precision_70:.2%}")

    # Lógica de Ação
    def classificar_acao(risco):
        if risco >= 0.90: return "1 - PRONTIDÃO CRÍTICA"
        elif risco >= limiar_otimizado: return "2 - INSPEÇÃO PREDITIVA"
        elif risco >= 0.40: return "3 - PREVENTIVA FOCADA"
        else: return "4 - MONITORAMENTO"

    ranking_rotas["Ação Recomendada"] = ranking_rotas["prob_risco"].apply(classificar_acao)

    # Exibição do Ranking
    st.subheader("📌 Ranking de Risco Gerado")
    df_exibir = ranking_rotas.sort_values("prob_risco", ascending=False)
    st.dataframe(
        df_exibir[["Rota_Afetada", "UF", "prob_risco", "Ação Recomendada"]]
        .style.apply(color_coluna, subset=['prob_risco'], axis=1)
        .format({"prob_risco": "{:.2%}"}),
        use_container_width=True
    )

    # --- PREPARAÇÃO PARA SALVAMENTO ---
    df_save = ranking_rotas.copy()
    data_ref = datetime.now().date() - timedelta(days=1)
    
    # Formato ISO string para o CSV não bugar (YYYY-MM-DD)
    df_save['Data'] = data_ref.strftime('%Y-%m-%d')
    df_save['ID'] = df_save['Rota_Afetada'].astype(str) + '_' + data_ref.strftime('%Y%m%d')
    
    df_save = df_save[[
        "ID", "Rota_Afetada", "UF", "prob_risco", "Ação Recomendada",
        "Frequencia_Mensal_Media", "Rota_Em_Obra_Selecionada", "Data"
    ]]
    df_save.columns = ["ID", "Rota", "UF", "Risco", "Action", "Frequencia", "Obra", "Data"]

    # Botão de Salvamento
    if st.button("Confirmar e Salvar no Histórico", type="secondary"):
        f_path = r"data\historico_ranking.csv"
        existe = os.path.exists(f_path)
        df_save.to_csv(f_path, mode='a', index=False, header=not existe, sep=';', encoding='utf-8-sig')
        st.success(f"✅ Dados arquivados com sucesso!")
        st.session_state['df_feedback_base'] = df_save.copy()

else:
    st.info("Aguardando configuração para iniciar o cálculo.")
