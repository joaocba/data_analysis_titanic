import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS

def show(data):
    st.markdown("### Distribuição por Idade")

    st.markdown("""
    Nesta secção, vamos explorar a distribuição das idades dos passageiros do Titanic. 
    Poderá visualizar as estatísticas principais, a distribuição geral e por faixa etária, 
    além de obter perspetivas sobre os padrões de idade e a sua relevância para a análise de sobrevivência.
    """)

    age_stats = data['Age'].describe()
    age_mode = data['Age'].mode()[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas da Distribuição")
        stats_df = pd.DataFrame({
            'Estatística': [
                'Número de registos',
                'Média',
                'Desvio Padrão',
                'Mínimo',
                '25º Percentil',
                'Mediana',
                '75º Percentil',
                'Máximo',
                'Moda'
            ],
            'Valor': [
                f"{age_stats['count']:.0f}",
                f"{age_stats['mean']:.2f} anos",
                f"{age_stats['std']:.2f} anos",
                f"{age_stats['min']:.2f} anos",
                f"{age_stats['25%']:.2f} anos",
                f"{age_stats['50%']:.2f} anos",
                f"{age_stats['75%']:.2f} anos",
                f"{age_stats['max']:.2f} anos",
                f"{age_mode:.2f} anos"
            ]
        })
        st.table(stats_df)

    with col2:
        age_bins = [0, 18, 65, 100]
        age_labels = ['Criança (0-17)', 'Adulto (18-64)', 'Idoso (65+)']
        data['FaixaEtaria'] = pd.cut(data['Age'],
                                     bins=age_bins,
                                     labels=age_labels,
                                     right=False)

        age_distribution = data['FaixaEtaria'].value_counts().sort_index()
        age_pct = (age_distribution / len(data) * 100).round(1)

        st.markdown("#### Distribuição por Faixa Etária")
        dist_df = pd.DataFrame({
            'Faixa Etária': age_distribution.index,
            'Passageiros': age_distribution.values,
            'Percentagem': age_pct.values
        })
        st.table(dist_df)

    st.markdown("### Visualizações e Perspetivas")
    _show_visualizations(data)
    _show_insights()

def _show_visualizations(data):
    st.markdown("#### Visualizações da Distribuição de Idade")
    tab1, tab2 = st.tabs(["Distribuição Geral", "Por Faixa Etária"])

    with tab1:
        _plot_age_distribution(data)

    with tab2:
        _plot_age_groups(data)

def _plot_age_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    sns.histplot(data=data,
                 x='Age',
                 bins=30,
                 color=colors[0],
                 alpha=0.6,
                 kde=True,
                 line_kws={'color': colors[1], 'linewidth': 2},
                 edgecolor='white',
                 linewidth=1)

    plt.axvline(x=data['Age'].mean(),
                color=colors[2],
                linestyle='--',
                linewidth=2,
                label=f'Média: {data["Age"].mean():.1f} anos')

    plt.axvline(x=data['Age'].median(),
                color=colors[4],
                linestyle='--',
                linewidth=2,
                label=f'Mediana: {data["Age"].median():.1f} anos')

    plt.title('Distribuição das Idades dos Passageiros do Titanic',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Idade (anos)', fontsize=12)
    plt.ylabel('Número de Passageiros', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    st.pyplot(fig)

def _plot_age_groups(data):
    age_distribution = data['FaixaEtaria'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    pie_colors = [colors[0], colors[2], colors[4]]

    plt.pie(age_distribution,
            labels=age_distribution.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            explode=(0.05, 0, 0.05))

    plt.title('Distribuição de Passageiros por Faixa Etária',
              pad=20, fontsize=14, fontweight='bold')
    st.pyplot(fig)

def _show_insights():
    st.divider()
    st.markdown("### Principais Perspetivas sobre a Idade")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Estatísticas Principais**")
        st.markdown("""
        - **Média de idade**: 29,32 anos
        - **Mediana**: 28,00 anos
        - A proximidade entre média e mediana indica uma **distribuição relativamente simétrica**.
        - **Desvio padrão**: 12,98 anos, indicando **variabilidade moderada nas idades**.
        """)

    with col2:
        st.markdown("#### **Análise por Faixas Etárias**")
        st.markdown("""
        - **Adultos (18-64)**: 86,1% dos passageiros
            - Representa a **grande maioria**.
            - Concentração entre **20-40 anos**.
        - **Crianças (0-17)**: 12,7% dos passageiros
            - **Proporção significativa**.
        - **Idosos (65+)**: 1,2% dos passageiros
            - **Presença muito reduzida**.
        """)

    st.markdown("#### **Implicações**")
    st.markdown("""
    - **Perfil maioritariamente jovem-adulto**.
    - **Possível viagem relacionada com migração/trabalho**.
    - **Presença significativa de famílias com crianças**.
    - **Padrões de viagem da época** refletidos na baixa presença de idosos.
    """)

    st.divider()

    st.markdown("### 📝 Resumo das Perspetivas")
    st.markdown("""
    **Perspetivas Importantes:**
    - O **perfil etário** dos passageiros é **maioritariamente jovem-adulto**, com uma forte presença de **adultos entre 18-64 anos**.
    - A presença de **crianças** (12,7%) sugere que muitas famílias estavam a viajar, enquanto a presença de **idosos** foi muito reduzida.
    - A **média e mediana** próximas indicam uma distribuição **relativamente simétrica**, com variação moderada (desvio padrão de 12,98 anos).
    - **Viagens associadas à migração ou trabalho** parecem ser um fator predominante entre os passageiros.
    """)

    st.divider()

    st.markdown("### 🔍 Detalhes sobre as Viagens")
    st.markdown("""
    **Possível Contexto Social:**
    - As famílias com crianças podem indicar que muitas destas viagens tinham caráter familiar ou de mudança.
    - A baixa presença de **idosos** pode sugerir que as condições da época dificultavam viagens para esta faixa etária.
    """)

    st.divider()