import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style

def show(data):
    st.markdown("### Distribuição por Dimensão de Família")

    st.markdown("""
    Nesta secção, vamos explorar a distribuição da dimensão das famílias dos passageiros do Titanic. 
    Poderá visualizar as estatísticas principais, a distribuição geral e proporções por diferentes dimensões de família, 
    além de obter perspetivas sobre os padrões de viagem e as suas implicações.
    """)

    family_stats = data['FamilySize'].describe()
    most_common_family_size = data['FamilySize'].mode()[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas da Distribuição")
        stats_df = pd.DataFrame({
            'Estatística': [
                'Número de registos',
                'Média de familiares',
                'Desvio Padrão',
                'Dimensão mínima de família',
                '25º Percentil',
                'Mediana',
                '75º Percentil',
                'Dimensão máxima de família',
                'Dimensão mais comum'
            ],
            'Valor': [
                f"{family_stats['count']:.0f}",
                f"{family_stats['mean']:.2f} pessoas",
                f"{family_stats['std']:.2f} pessoas",
                f"{family_stats['min']:.2f} pessoas",
                f"{family_stats['25%']:.2f} pessoas",
                f"{family_stats['50%']:.2f} pessoas",
                f"{family_stats['75%']:.2f} pessoas",
                f"{int(family_stats['max'])} pessoas",
                f"{int(most_common_family_size)} pessoas"
            ]
        })
        st.table(stats_df)

    with col2:
        family_dist = data['FamilySize'].value_counts().sort_index()
        family_pct = (family_dist / len(data) * 100).round(1)

        st.markdown("#### Distribuição por Dimensão de Família")
        dist_df = pd.DataFrame({
            'Dimensão de Família': family_dist.index,
            'Passageiros': family_dist.values,
            'Percentagem': family_pct.values
        })
        st.table(dist_df)

    st.markdown("### Visualizações e Perspetivas")
    _show_visualizations(data, family_dist)
    _show_insights(data)

def _show_visualizations(data, family_dist):
    st.markdown("#### Visualizações")
    tab1, tab2 = st.tabs(["Distribuição Geral", "Proporção Sozinho vs. Família"])

    with tab1:
        _plot_family_distribution(family_dist)

    with tab2:
        _plot_alone_vs_family(data)

def _plot_family_distribution(family_dist):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = get_color_palette(20)

    sns.barplot(x=family_dist.index,
                y=family_dist.values,
                color=colors[0])

    set_plot_style(
        ax,
        'Distribuição da Dimensão das Famílias',
        'Dimensão da Família',
        'Número de Passageiros'
    )

    for i, v in enumerate(family_dist.values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    st.pyplot(fig)

def _plot_alone_vs_family(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_color_palette(20)

    alone_vs_family = data['IsAlone'].value_counts()

    plt.pie(alone_vs_family.values,
            labels=['Sozinho', 'Com Família'],
            autopct='%1.1f%%',
            colors=[colors[0], colors[2]],
            explode=(0.05, 0))

    plt.title('Proporção de Passageiros: Sozinhos vs. Com Família',
              pad=20, fontsize=14, fontweight='bold')
    st.pyplot(fig)

def _show_insights(data):
    st.divider()
    st.markdown("### Principais Perspetivas sobre a Dimensão das Famílias")

    alone_pct = (data['IsAlone'].sum() / len(data) * 100)
    small_family = ((data['FamilySize'] > 0) & (data['FamilySize'] <= 2)).sum()
    small_family_pct = (small_family / len(data) * 100)
    large_family = (data['FamilySize'] > 2).sum()
    large_family_pct = (large_family / len(data) * 100)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas Gerais")
        st.markdown(f"""
        - **Média de familiares por passageiro**: {data['FamilySize'].mean():.2f}
        - **Dimensão máxima de família**: {int(data['FamilySize'].max())} pessoas
        - **Desvio padrão**: {data['FamilySize'].std():.2f}
        - **A viajar sozinhos**: {alone_pct:.2f}% dos passageiros
        """)

    with col2:
        st.markdown("#### Padrões de Viagem")
        st.markdown("""
        - **Famílias pequenas (1-2 pessoas)**: Representam uma grande parcela dos passageiros.
        - **Famílias grandes (3+ pessoas)**: Mais comuns na 3ª classe.
        - **Proporção de viagens individuais** é alta, especialmente entre passageiros da 3ª classe.
        """)

    st.markdown("""
    **Implicações Socioeconómicas**
    - Famílias pequenas podem indicar viagens familiares, enquanto famílias grandes sugerem deslocações em grupo.
    - A alta proporção de passageiros sozinhos reflete padrões de viagem típicos de migração ou trabalho.
    """)

    st.divider()