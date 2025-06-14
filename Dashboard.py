import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------Page Configs

st.set_page_config(layout='wide')

# -----------------------------
#           Funções
#------------------------------
def formataNumero(valor):
    if valor >= 1_000_000_000:
        return f'{valor / 1_000_000_000:.1f} b ' 
    if valor >= 1_000_000:
        return f'{valor / 1_000_000:.1f} m '
    if valor >= 1000:
        return f'{valor / 1000:.1f} k '
    return str(valor)
    

# -----------------------------
#           Dados
#------------------------------

df = pd.read_csv ('smart_manufacturing_data.csv')

# -----------------------------
#         Imputação
#------------------------------

numeric_cols = ['temperature', 'vibration',  'pressure', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

categorical_cols = ['failure_type', 'downtime_risk', 'maintenance_required']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

condition = (df['machine_status'].isnull()) & (df['anomaly_flag'] == 'Yes') & (df['maintenance_required'] == 'Yes')
df.loc[condition, 'machine_status'] = 'Failure'

vibration_threshold = df['vibration'].quantile(0.9)
temp_threshold = df['temperature'].quantile(0.9)
condition = (df['machine_status'].isnull()) & (df['vibration'] > vibration_threshold) & (df['temperature'] > temp_threshold)
df.loc[condition, 'machine_status'] = 'Warning'

for machine in df['machine'].unique():
    machine_mask = (df['machine'] == machine) & (df['machine_status'].isnull())
    most_common = df[df['machine'] == machine]['machine_status'].mode()
    if not most_common.empty:
        df.loc[machine_mask, 'machine_status'] = most_common[0]

# ================================
# Barra Lateral - Filtros Globais
# ================================
st.sidebar.header("Filtros")

# Filtro 1: Por Máquina
machines = df['machine'].dropna().unique()
selected_machine = st.sidebar.multiselect("Selecionar Máquina(s):", machines, default=machines[:3])

# Filtro 2: Por Status da Máquina
statuses = df['machine_status'].dropna().unique()
selected_status = st.sidebar.multiselect("Selecionar Status da Máquina:", statuses, default=statuses)

# Filtro 3: Por Período (Timestamp)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
selected_date = st.sidebar.date_input("Selecionar Período:", [min_date, max_date])

# Aplicar os filtros
filtered_df = df[
    (df['machine'].isin(selected_machine)) &
    (df['machine_status'].isin(selected_status)) &
    (df['timestamp'].between(pd.to_datetime(selected_date[0]), pd.to_datetime(selected_date[1])))
]

# ================================
# Página Principal - Abas
# ================================
st.title("Dashboard de Manufatura Inteligente")

tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "🛠️ Manutenção", "⚠️ Riscos & Vida Útil"])

# ===================================
# Aba 1: Visão Geral
# ===================================
with tab1:
    st.header("Distribuições e Status das Máquinas")

    # Gráfico 1: Status das Máquinas (Pizza)
    status_counts = filtered_df['machine_status'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title("Status das Máquinas")
    st.pyplot(fig)

    # Gráfico 2: Distribuição da Temperatura
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['temperature'].dropna(), kde=True, ax=ax, color='orange')
    ax.set_title("Distribuição de Temperatura")
    st.pyplot(fig)

    
    
    # Gráfico 3: Consumo Médio de Energia ao Longo do Tempo (com agrupamento condicional)
    energy_df = filtered_df.copy()
    energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])

    # Calcula o intervalo de tempo filtrado
    min_date = energy_df['timestamp'].min()
    max_date = energy_df['timestamp'].max()
    date_range = (max_date - min_date).days

    fig, ax = plt.subplots(figsize=(10, 4))

    if date_range > 2:
        # Se o período for maior que 2 dias, agrupa por dia
        energy_df['data'] = energy_df['timestamp'].dt.date
        daily_energy = energy_df.groupby('data')['energy_consumption'].mean().reset_index()

        ax.plot(daily_energy['data'], daily_energy['energy_consumption'], color='blue', marker='o', linestyle='-')
        ax.set_title(f"Consumo Médio de Energia por Dia ({min_date.date()} a {max_date.date()})")
        ax.set_xlabel("Data")
    else:
        # Se for até 2 dias, plota o dado original (sem agregação)
        ax.plot(energy_df['timestamp'], energy_df['energy_consumption'], color='blue', marker='o', linestyle='-')
        ax.set_title(f"Consumo de Energia ao Longo do Tempo ({min_date.date()} a {max_date.date()})")
        ax.set_xlabel("Timestamp")

    ax.set_ylabel("Consumo Médio de Energia")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ===================================
# Aba 2: Análise de Manutenção
# ===================================
with tab2:
    st.header("Análise de Manutenção e Anomalias")

    # Gráfico 4: Máquinas que Precisam de Manutenção
    maintenance_needed = filtered_df[filtered_df['maintenance_required'] == 'Yes']
    machines_needing_maintenance = maintenance_needed['machine'].nunique()
    st.metric("Máquinas Precisando de Manutenção", machines_needing_maintenance)

    # Gráfico 5: Anomalias por Tipo de Falha (excluindo 'Operação Normal')
    falhas_df = filtered_df[filtered_df['failure_type'] != 'Normal']

    fig, ax = plt.subplots()
    sns.countplot(data=falhas_df, x='failure_type', ax=ax, palette='Set2')
    ax.set_title("Anomalias por Tipo de Falha")
    st.pyplot(fig)


    # Gráfico 6: Correlação entre Variáveis (Mapa de Calor)
    numeric_features = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']
    corr_matrix = filtered_df[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Mapa de Calor - Correlação entre Variáveis ")
    st.pyplot(fig)

# ===================================
# Aba 3: Riscos e Vida Útil
# ===================================
with tab3:
    st.header("Análise de Riscos e Vida Útil")

    # Gráfico 7: Distribuição de Downtime Risk (Histograma + KDE)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df['downtime_risk'].dropna(), bins=30, kde=True, color='purple', ax=ax)
    ax.set_title("Distribuição de Downtime Risk")
    ax.set_xlabel("Downtime Risk")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

    # Gráfico 8: Predicted Remaining Life por Máquina
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, y='machine', x='predicted_remaining_life', 
                hue='machine_status', ax=ax, palette='Set2')
    ax.set_title("Distribuição da Vida Útil Prevista por Máquina")
    ax.set_xlabel("Vida Útil Prevista (horas/dias)")
    ax.set_ylabel("Máquina")
    min_life = filtered_df['predicted_remaining_life'].min()
    machine_at_risk = filtered_df.loc[filtered_df['predicted_remaining_life'].idxmin(), 'machine']

    col1, col2 = st.columns(2)
    col1.metric("Menor Vida Útil Prevista", f"{min_life} horas/dias")
    col2.metric("Máquina com Maior Risco", machine_at_risk)
    st.pyplot(fig)

    # Gráfico 9: Temperatura vs Vibração
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x='temperature', y='vibration', hue='machine_status', ax=ax)
    ax.set_title("Temperatura vs Vibração")
    st.pyplot(fig)

# ===================================
# Nova Página: Dados + Download
# ===================================
with st.expander("📥 Visualizar e Baixar os Dados Filtrados"):
    st.dataframe(filtered_df)

    # Botão para download
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar CSV",
        data=csv,
        file_name='dados_filtrados.csv',
        mime='text/csv',
    )
