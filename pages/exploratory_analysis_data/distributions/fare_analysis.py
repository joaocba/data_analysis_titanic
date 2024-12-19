import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS

def show(data):
    st.markdown("### Distribuição por Tarifas")

    st.markdown("""
    Nesta secção, vamos analisar as tarifas pagas pelos passageiros do Titanic. 
    Veremos as estatísticas principais, a distribuição geral e por classe, além de identificar perspetivas 
    relacionadas com os padrões de tarifação.
    """)

    fare_stats = data['Fare'].describe()
    fare_mode = data['Fare'].mode()[0]

    col1, col2 = st.columns(2)

    with col1:
        _show_general_stats(fare_stats, fare_mode)

    with col2:
        _show_class_stats(data)

    st.markdown("### Visualizações e Perspetivas")
    _show_visualizations(data)
    _show_insights()

def _show_general_stats(fare_stats, fare_mode):
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
            f"{fare_stats['count']:.0f}",
            f"£{fare_stats['mean']:.2f}",
            f"£{fare_stats['std']:.2f}",
            f"£{fare_stats['min']:.2f}",
            f"£{fare_stats['25%']:.2f}",
            f"£{fare_stats['50%']:.2f}",
            f"£{fare_stats['75%']:.2f}",
            f"£{fare_stats['max']:.2f}",
            f"£{fare_mode:.2f}"
        ]
    })
    st.table(stats_df)

def _show_class_stats(data):
    st.markdown("#### Análise por Classe")
    class_stats = pd.DataFrame()

    for pclass in [1, 2, 3]:
        stats = data[data['Pclass'] == pclass]['Fare'].describe()
        pct_passengers = (len(data[data['Pclass'] == pclass]) / len(data) * 100)
        class_stats[f'{pclass}ª Classe'] = [
            len(data[data['Pclass'] == pclass]),
            f"{pct_passengers:.1f}%",
            f"£{stats['mean']:.2f}",
            f"£{stats['50%']:.2f}",
            f"£{stats['max']:.2f}"
        ]

    class_stats.index = ['Passageiros', '% do Total', 'Média', 'Mediana', 'Máximo']

    class_stats = class_stats.map(str)
    st.table(class_stats)

def _show_visualizations(data):
    st.markdown("#### Visualizações")
    tab1, tab2, tab3 = st.tabs([
        "Distribuição Geral",
        "Estatísticas por Classe",
        "Densidade por Classe"
    ])

    with tab1:
        _plot_fare_distribution(data)

    with tab2:
        _plot_class_statistics(data)

    with tab3:
        _plot_class_density(data)

def _plot_fare_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 5, 20))

    sns.histplot(data=data,
                 x='Fare',
                 bins=50,
                 color=colors[0],
                 alpha=0.7)

    plt.axvline(x=data['Fare'].mean(),
                color=colors[2],
                linestyle='--',
                linewidth=2,
                label=f'Média: £{data["Fare"].mean():.2f}')

    plt.axvline(x=data['Fare'].median(),
                color=colors[4],
                linestyle='--',
                linewidth=2,
                label=f'Mediana: £{data["Fare"].median():.2f}')

    plt.title('Distribuição das Tarifas',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Tarifa (£)', fontsize=12)
    plt.ylabel('Número de Passageiros', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def _plot_class_statistics(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(3)
    width = 0.25
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    stats_by_class = data.groupby('Pclass')['Fare'].agg(['mean', 'median', 'max']).round(2)

    plt.bar(x - width, stats_by_class['mean'],
            width, label='Média',
            color=colors[0])
    plt.bar(x, stats_by_class['median'],
            width, label='Mediana',
            color=colors[2])
    plt.bar(x + width, stats_by_class['max'],
            width, label='Máximo',
            color=colors[4])

    plt.title('Estatísticas de Tarifas por Classe',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel('Tarifa (£)', fontsize=12)
    plt.xticks(x, ['1ª Classe', '2ª Classe', '3ª Classe'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def _plot_class_density(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette('Set2', n_colors=3)

    for pclass, color in zip([1, 2, 3], colors):
        sns.kdeplot(
            data=data[data['Pclass'] == pclass]['Fare'],
            label=f'{pclass}ª Classe',
            color=color,
            fill=True,
            alpha=0.5
        )

    plt.title('Distribuição de Densidade das Tarifas por Classe',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarifa (£)', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.legend(title='Classes')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def _show_insights():
    st.divider()
    st.markdown("### Principais Perspetivas sobre as Tarifas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Estatísticas Principais**")
        st.markdown("""
        - **Tarifa média**: £32,10
        - **Tarifa mediana**: £14,45
        - A grande diferença entre média e mediana sugere uma **distribuição assimétrica**.
        - **Amplitude**: de £0,00 a £512,33, com uma **grande variação** nas tarifas.
        """)

    with col2:
        st.markdown("#### **Análise por Classe**")
        st.markdown("""
        - **1ª Classe**: Tarifa média de **£84,19** (24,1% dos passageiros), com grande variabilidade nos preços.
        - **2ª Classe**: Tarifa média de **£20,66** (20,7% dos passageiros), com preços mais consistentes.
        - **3ª Classe**: Tarifa média de **£13,68** (55,2% dos passageiros), com menor variabilidade.
        """)

    st.markdown("#### **Implicações Socioeconómicas**")
    st.markdown("""
    - **Estratificação social** evidente, com tarifas na **1ª Classe** sendo cerca de **6 vezes mais caras** que as da **3ª Classe**.
    - A maioria dos passageiros está na classe mais económica (**55,2% na 3ª Classe**).
    - A desigualdade socioeconómica entre passageiros é clara.
    """)

    st.divider()

    st.markdown("### 📝 Resumo das Perspetivas")
    st.markdown("""
    - **Distribuição assimétrica** das tarifas, com uma **média maior que a mediana**.
    - **Maioria dos passageiros na 3ª Classe**, refletindo a natureza do Titanic como um meio de transporte para pessoas de diferentes classes sociais.
    """)

    st.divider()