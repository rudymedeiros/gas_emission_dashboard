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

dados = pd.read_csv ('emissoes.csv')


# -----------------------------
#           Tabelas
#------------------------------

emissoes_estados = dados.groupby('Estado')[['Emissão']].sum().reset_index()
emissoes_estados = dados.drop_duplicates(subset='Estado')[['Estado', 'lat', 'long']].merge(emissoes_estados, on='Estado').reset_index()
emissoes_estados.drop('index', axis=1, inplace=True)

# -----------------------------
#           Gráfico
#------------------------------

fig_mapa_emissoes = px.scatter_geo(emissoes_estados,
                                   lat = 'lat',
                                   lon='long',
                                   scope='south america',
                                   hover_name='Estado',
                                   hover_data={'lat':False,'long':False},
                                   size='Emissão',
                                   color ='Estado',
                                   text='Estado',
                                   title='Total de Emissões por estado'
                                   )

# -----------------------------
#           Dashboard
#------------------------------

st.title("Emissões Gases Efeito Estufa")

st.metric("Total de Emissões",formataNumero(dados['Emissão'].sum()) +  'de toneladas')
st.plotly_chart(fig_mapa_emissoes)

st.dataframe(dados)