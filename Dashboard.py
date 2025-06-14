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

df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Imputação para colunas numéricas
numeric_cols = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']
for col in numeric_cols:
    if col in df.columns:
        # Preenche com a mediana (menos sensível a outliers) agrupada por máquina
        df[col] = df.groupby('machine')[col].transform(lambda x: x.fillna(x.median()))
        # Se ainda houver NAs (caso toda o grupo seja NA), preenche com a mediana global
        df[col] = df[col].fillna(df[col].median())

# 2. Imputação para colunas categóricas
categorical_cols = ['failure_type', 'maintenance_required', 'anomaly_flag', 'machine_status']
for col in categorical_cols:
    if col in df.columns:
        # Preenche com a moda agrupada por máquina
        df[col] = df.groupby('machine')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))
        # Preenche quaisquer valores restantes com 'Unknown'
        df[col] = df[col].fillna('Unknown')

# 3. Imputação inteligente do machine_status baseado em outras variáveis
# Primeiro cria uma máscara para os status ainda faltantes
missing_status = df['machine_status'].isna()

# Regra 1: Se anomaly_flag é 'Yes' e maintenance_required é 'Yes', provavelmente é 'Failure'
df.loc[missing_status & (df['anomaly_flag'] == 'Yes') & (df['maintenance_required'] == 'Yes'), 'machine_status'] = 'Failure'

# Regra 2: Valores extremos de vibração e temperatura indicam 'Warning'
vibration_threshold = df['vibration'].quantile(0.9)
temp_threshold = df['temperature'].quantile(0.9)
df.loc[missing_status & 
       (df['vibration'] > vibration_threshold) & 
       (df['temperature'] > temp_threshold), 'machine_status'] = 'Warning'

# Regra 3: Para os demais casos, usa o status mais comum daquela máquina
for machine in df[df['machine_status'].isna()]['machine'].unique():
    most_common = df[df['machine'] == machine]['machine_status'].mode()
    if not most_common.empty:
        df.loc[(df['machine'] == machine) & (df['machine_status'].isna()), 'machine_status'] = most_common[0]

# Regra 4: Qualquer valor ainda faltante recebe 'Running' como padrão
df['machine_status'] = df['machine_status'].fillna('Running')


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

    last_status_per_machine = filtered_df.sort_values('timestamp').groupby('machine').last()
    current_maintenance_needed = last_status_per_machine[last_status_per_machine['maintenance_required'] == 'Yes']
    machines_needing_maintenance = current_maintenance_needed.shape[0]
    st.metric("Máquinas Atualmente Precisando de Manutenção", machines_needing_maintenance)
    if not current_maintenance_needed.empty:
        st.write("**Máquinas com manutenção pendente:**")
        st.dataframe(current_maintenance_needed[['machine_status', 'failure_type', 'predicted_remaining_life']])
    else:
        st.success("Nenhuma máquina requer manutenção atualmente!")

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
    
    # Seção melhorada de Downtime Risk
    st.subheader("Análise de Downtime Risk")
    
    # 1. Métricas-chave
    col1, col2, col3 = st.columns(3)
    col1.metric("Média Downtime Risk", f"{filtered_df['downtime_risk'].mean():.2f}")
    col2.metric("Máximo Downtime Risk", f"{filtered_df['downtime_risk'].max():.2f}")
    col3.metric("Máquinas em Alto Risco", 
                f"{(filtered_df['downtime_risk'] > 0.7).sum()} ({(filtered_df['downtime_risk'] > 0.7).mean()*100:.1f}%)")
    
    # 2. Visualização adaptativa
    tab_risk1, tab_risk2, tab_risk3 = st.tabs(["Distribuição", "Por Máquina", "Tendência Temporal"])
    
    with tab_risk1:
        # Gráfico de distribuição melhorado
        fig = plt.figure(figsize=(10, 6))
        
        # Usamos bins especiais para destacar 0, 1 e valores intermediários
        bins = [-0.05, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]
        plt.hist(filtered_df['downtime_risk'], bins=bins, edgecolor='black', alpha=0.7)
        
        plt.title("Distribuição de Downtime Risk (Bins Especiais)")
        plt.xlabel("Nível de Risco")
        plt.ylabel("Frequência")
        plt.xticks([0, 0.5, 1], ['0 (Baixo)', '0.5 (Médio)', '1 (Alto)'])
        st.pyplot(fig)
        
        st.caption("""
        **Interpretação:**  
        - Valores próximos a 0: Risco mínimo de parada  
        - Valores próximos a 1: Alto risco de parada  
        - Distribuição bimodal sugere classificações binárias
        """)
    
    with tab_risk2:
        # Top 10 máquinas com maior risco
        high_risk = filtered_df.groupby('machine')['downtime_risk'].max().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        high_risk.sort_values().plot(kind='barh', color='coral', ax=ax)
        plt.title("Top 10 Máquinas com Maior Downtime Risk")
        plt.xlabel("Máximo Downtime Risk")
        plt.ylabel("Máquina")
        st.pyplot(fig)
    
    with tab_risk3:
        # Tendência temporal (apenas se o filtro cobrir múltiplas datas)
        if filtered_df['timestamp'].nunique() > 1:
            fig = px.line(filtered_df.groupby(pd.Grouper(key='timestamp', freq='D'))['downtime_risk'].mean(),
                         title="Evolução Diária do Downtime Risk Médio",
                         labels={'value': 'Downtime Risk Médio', 'timestamp': 'Data'})
            st.plotly_chart(fig)
        else:
            st.warning("Selecione um intervalo de tempo maior para visualizar a tendência temporal")

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
