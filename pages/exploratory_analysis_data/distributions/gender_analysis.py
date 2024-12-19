import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage

def show(data):
    st.markdown("### Distribuição por Sexo")

    st.markdown("""
    Nesta secção, vamos explorar a distribuição por sexo dos passageiros do Titanic. 
    Poderá visualizar as estatísticas principais, a distribuição geral e por classe social, 
    além de obter perspetivas sobre os padrões de género e a sua relevância para a análise de sobrevivência.
    """)

    sex_dist = data['Sex'].value_counts()
    sex_pct = (sex_dist / len(data) * 100)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas da Distribuição")
        stats_df = pd.DataFrame({
            'Estatística': [
                'Total de passageiros',
                'Homens',
                'Mulheres',
                'Proporção H/M',
                'Percentagem Masculina',
                'Percentagem Feminina'
            ],
            'Valor': [
                f"{len(data)}",
                f"{sex_dist[0]}",
                f"{sex_dist[1]}",
                f"{(sex_dist[0] / sex_dist[1]):.2f}",
                f"{format_percentage(sex_pct[0])}",
                f"{format_percentage(sex_pct[1])}"
            ]
        })
        st.table(stats_df)

    with col2:
        st.markdown("#### Distribuição por Classe")
        class_sex_dist = pd.crosstab(data['Pclass'], data['Sex'])
        class_sex_pct = pd.crosstab(data['Pclass'], data['Sex'], normalize='index') * 100

        class_dist_df = pd.DataFrame({
            'Classe': ['1ª Classe', '2ª Classe', '3ª Classe'],
            'Total': [
                sum(class_sex_dist.loc[1]),
                sum(class_sex_dist.loc[2]),
                sum(class_sex_dist.loc[3])
            ],
            'Homens (%)': [
                f"{class_sex_dist.loc[1, 0]} ({format_percentage(class_sex_pct.loc[1, 0])})",
                f"{class_sex_dist.loc[2, 0]} ({format_percentage(class_sex_pct.loc[2, 0])})",
                f"{class_sex_dist.loc[3, 0]} ({format_percentage(class_sex_pct.loc[3, 0])})"
            ],
            'Mulheres (%)': [
                f"{class_sex_dist.loc[1, 1]} ({format_percentage(class_sex_pct.loc[1, 1])})",
                f"{class_sex_dist.loc[2, 1]} ({format_percentage(class_sex_pct.loc[2, 1])})",
                f"{class_sex_dist.loc[3, 1]} ({format_percentage(class_sex_pct.loc[3, 1])})"
            ]
        })
        st.table(class_dist_df)

    st.markdown("### Visualizações e Perspetivas")
    _show_visualizations(data)
    _show_insights(data)

def _show_visualizations(data):
    st.markdown("#### Visualizações da Distribuição por Sexo")
    tab1, tab2 = st.tabs(["Distribuição Geral", "Por Classe Social"])

    with tab1:
        _plot_gender_distribution(data)

    with tab2:
        _plot_class_distribution(data)

def _plot_gender_distribution(data):
    sex_dist = data['Sex'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_color_palette(20)

    plt.pie(sex_dist.values,
            labels=['Masculino', 'Feminino'],
            autopct=lambda pct: f'{format_percentage(pct)}\n({int(pct * len(data) / 100)})',
            colors=[colors[0], colors[2]],
            explode=(0.05, 0.05),
            startangle=90)

    plt.title('Distribuição de Passageiros por Sexo',
              pad=20, fontsize=14, fontweight='bold')
    st.pyplot(fig)

def _plot_class_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    class_sex_dist = pd.crosstab(data['Pclass'], data['Sex'])
    colors = get_color_palette(20)

    x = np.arange(len(['1ª Classe', '2ª Classe', '3ª Classe']))
    width = 0.35

    plt.bar(x - width / 2, class_sex_dist[0],
            width, label='Masculino',
            color=colors[0])
    plt.bar(x + width / 2, class_sex_dist[1],
            width, label='Feminino',
            color=colors[2])

    set_plot_style(
        ax,
        'Distribuição por Sexo por Classe Social',
        'Classe',
        'Número de Passageiros'
    )
    plt.xticks(x, ['1ª Classe', '2ª Classe', '3ª Classe'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    for i in range(len(x)):
        plt.text(i - width / 2, class_sex_dist[0][i + 1],
                 str(class_sex_dist[0][i + 1]),
                 ha='center', va='bottom')
        plt.text(i + width / 2, class_sex_dist[1][i + 1],
                 str(class_sex_dist[1][i + 1]),
                 ha='center', va='bottom')

    st.pyplot(fig)

def _show_insights(data):
    st.divider()
    st.markdown("### Principais Perspetivas sobre o Sexo dos Passageiros")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Estatísticas Principais**")
        total_men = len(data[data['Sex'] == 0])
        total_women = len(data[data['Sex'] == 1])
        st.markdown(f"""
        - **Total de homens**: {total_men} ({format_percentage(total_men / len(data))}%)
        - **Total de mulheres**: {total_women} ({format_percentage(total_women / len(data))}%)
        - **Razão H/M**: {(total_men / total_women):.2f} homens para cada mulher
        - **Predominância**: Masculina em todas as classes
        """)

    with col2:
        st.markdown("#### **Análise por Classes**")
        class_sex_pct = pd.crosstab(data['Pclass'], data['Sex'], normalize='index') * 100
        st.markdown(f"""
        - **1ª Classe**: Distribuição mais equilibrada
            - Homens: {format_percentage(class_sex_pct.loc[1, 0])}
            - Mulheres: {format_percentage(class_sex_pct.loc[1, 1])}
        - **2ª Classe**: Semelhante à primeira classe
        - **3ª Classe**: Maior desequilíbrio
            - Predominância masculina acentuada
        """)

    st.divider()

    st.markdown("### 📊 Implicações Sociais e Históricas")
    st.markdown("""
    **Padrões Observados:**
    - **Estrutura Social**: Classes mais altas apresentam distribuição mais equilibrada entre sexos
    - **Padrões Migratórios**: Forte presença masculina na 3ª classe sugere migração económica
    - **Aspetos Económicos**: Diferenças na distribuição refletem realidade socioeconómica da época
    """)

    st.divider()

    st.markdown("### 🔍 Contexto Histórico")
    st.markdown("""
    **Relevância Histórica:**
    - A distribuição por sexo reflete os **padrões de migração** do início do século XX
    - As diferenças entre classes mostram a **estratificação social** da época
    - A predominância masculina na 3ª classe sugere um perfil de **migração económica**
    """)

    st.divider()