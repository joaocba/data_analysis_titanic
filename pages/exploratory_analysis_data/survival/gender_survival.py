import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise de sobrevivência por género"""
    st.markdown("### Análise de Sobrevivência por Género")

    st.markdown("""
    Nesta secção, iremos explorar como o género influenciou as hipóteses de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as taxas de sobrevivência por género, compreender os padrões de sobrevivência 
    entre diferentes classes sociais e identificar como as normas sociais da época influenciaram estes resultados.
    """)

    # Preparar dados
    survival_by_sex = pd.crosstab(data['Sex'], data['Survived'])
    survival_rate_sex = pd.crosstab(data['Sex'], data['Survived'], normalize='index') * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas por Género")
        stats_df = _create_gender_stats_df(data)
        st.table(stats_df)

    with col2:
        st.markdown("#### Taxa de Sobrevivência")
        survival_stats_df = _create_survival_stats_df(survival_by_sex, survival_rate_sex)
        st.table(survival_stats_df)

    # Visualizações e Conclusões
    st.markdown("### Visualizações e Conclusões")
    _show_visualizations(data, survival_by_sex)
    _show_insights(data)


def _create_gender_stats_df(data):
    """Cria DataFrame com estatísticas por género"""
    genders = {0: 'Masculino', 1: 'Feminino'}
    stats = []

    for gender_code, gender_name in genders.items():
        gender_data = data[data['Sex'] == gender_code]
        stats.append({
            'Género': gender_name,
            'Total': len(gender_data),
            'Idade Média': f"{gender_data['Age'].mean():.1f} anos",
            'Classes Mais Comuns': f"{gender_data['Pclass'].mode().iloc[0]}.ª Classe",
            'Proporção': format_percentage(len(gender_data) / len(data) * 100)
        })

    return pd.DataFrame(stats)


def _create_survival_stats_df(survival_by_sex, survival_rate_sex):
    """Cria DataFrame com estatísticas de sobrevivência"""
    return pd.DataFrame({
        'Género': ['Masculino', 'Feminino'],
        'Total': [survival_by_sex.loc[0].sum(), survival_by_sex.loc[1].sum()],
        'Sobreviventes': [survival_by_sex.loc[0, 1], survival_by_sex.loc[1, 1]],
        'Não Sobreviventes': [survival_by_sex.loc[0, 0], survival_by_sex.loc[1, 0]],
        'Taxa Sobrevivência': [
            format_percentage(survival_rate_sex.loc[0, 1]),
            format_percentage(survival_rate_sex.loc[1, 1])
        ]
    })


def _show_visualizations(data, survival_by_sex):
    """Apresenta visualizações da sobrevivência por género"""
    st.markdown("#### Visualizações da Sobrevivência por Género")
    tab1, tab2, tab3 = st.tabs(["Distribuição Geral", "Por Classe Social", "Análise Detalhada"])

    with tab1:
        _plot_survival_distribution(data, survival_by_sex)

    with tab2:
        _plot_class_distribution(data)

    with tab3:
        _plot_detailed_analysis(data)


def _plot_survival_distribution(data, survival_by_sex):
    """Apresenta a distribuição geral de sobrevivência por género"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = get_color_palette(20)

    x = np.arange(2)
    width = 0.35

    # Criar as barras
    non_survivors = plt.bar(x - width / 2, [survival_by_sex.loc[0, 0], survival_by_sex.loc[1, 0]],
                            width, label='Não Sobreviveu', color=colors[3])
    survivors = plt.bar(x + width / 2, [survival_by_sex.loc[0, 1], survival_by_sex.loc[1, 1]],
                        width, label='Sobreviveu', color=colors[0])

    set_plot_style(
        ax,
        'Distribuição de Sobrevivência por Género',
        'Género',
        'Número de Passageiros'
    )

    plt.xticks(x, ['Masculino', 'Feminino'])
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar valores nas barras (agora em preto)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., height / 2.,
                     f'{int(height)}',
                     ha='center', va='center', color='black', fontweight='bold')

    autolabel(non_survivors)
    autolabel(survivors)

    plt.tight_layout()
    st.pyplot(fig)


def _plot_class_distribution(data):
    """Apresenta a distribuição de sobrevivência por género e classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    class_gender_survival = pd.crosstab([data['Pclass'], data['Sex']], data['Survived'], normalize='index') * 100
    class_gender_survival = class_gender_survival[1].unstack()  # Obter apenas taxa de sobrevivência

    class_gender_survival.plot(kind='bar', ax=ax)

    set_plot_style(
        ax,
        'Taxa de Sobrevivência por Classe e Género',
        'Classe',
        'Taxa de Sobrevivência (%)'
    )

    # Atualizar rótulos do eixo x
    ax.set_xticklabels(['1.ª Classe', '2.ª Classe', '3.ª Classe'], rotation=0)
    plt.legend(['Masculino', 'Feminino'])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    st.pyplot(fig)


def _plot_detailed_analysis(data):
    """Apresenta análise detalhada por género"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular contagens e percentagens
    counts = pd.crosstab(data['Sex'], data['Survived'])
    percentages = pd.crosstab(data['Sex'], data['Survived'], normalize='index') * 100

    # Converter para arrays para facilitar o plotting
    non_survivors = counts[0].values
    survivors = counts[1].values
    non_survivors_pct = percentages[0].values
    survivors_pct = percentages[1].values

    # Apresentar barras empilhadas
    bottom_bars = ax.bar([0, 1], non_survivors,
                         label='Não Sobreviveu',
                         color=COLORS['negative'])
    top_bars = ax.bar([0, 1], survivors,
                      bottom=non_survivors,
                      label='Sobreviveu',
                      color=COLORS['primary'])

    set_plot_style(
        ax,
        'Composição de Sobrevivência por Género',
        'Género',
        'Número de Passageiros'
    )

    plt.xticks([0, 1], ['Masculino', 'Feminino'])
    plt.legend(title='Estado', bbox_to_anchor=(1.0, 1.02))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar percentagens em cada barra
    for idx, rect in enumerate(bottom_bars):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                height / 2.,
                f'{non_survivors_pct[idx]:.1f}%',
                ha='center', va='center',
                color='black', fontweight='bold')

    for idx, rect in enumerate(top_bars):
        height = rect.get_height()
        bottom = non_survivors[idx]
        ax.text(rect.get_x() + rect.get_width() / 2.,
                bottom + height / 2.,
                f'{survivors_pct[idx]:.1f}%',
                ha='center', va='center',
                color='black', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def _show_insights(data):
    """Apresenta as conclusões da análise de sobrevivência por género"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Género e Sobrevivência")

    # Dividindo as secções em colunas para melhor visualização
    col1, col2 = st.columns(2)

    # Estatísticas por Grupo
    with col1:
        st.markdown("#### **Análise por Género**")
        male_data = data[data['Sex'] == 0]
        female_data = data[data['Sex'] == 1]

        st.markdown(f"""
        - **Mulheres**:
            - Total: {len(female_data)}
            - Sobreviventes: {len(female_data[female_data['Survived'] == 1])}
            - Taxa: {format_percentage(female_data['Survived'].mean() * 100)}
        - **Homens**:
            - Total: {len(male_data)}
            - Sobreviventes: {len(male_data[male_data['Survived'] == 1])}
            - Taxa: {format_percentage(male_data['Survived'].mean() * 100)}
        - **Disparidade**:
            - Diferença de {format_percentage(abs(female_data['Survived'].mean() - male_data['Survived'].mean()) * 100)}
            - Forte influência do género na sobrevivência
        """)

    # Análise por Classe
    with col2:
        st.markdown("#### **Análise por Classe**")
        st.markdown("""
        - **1.ª Classe**:
            - Maior taxa de sobrevivência para ambos
            - Privilégio social evidente
            - Melhor acesso aos botes
        - **2.ª Classe**:
            - Taxas intermédias
            - Padrão similar à primeira classe
        - **3.ª Classe**:
            - Menores taxas de sobrevivência
            - Maior disparidade entre géneros
        """)

    st.divider()

    # Factores Sociais
    st.markdown("### 👥 Factores Sociais")
    st.markdown("""
    **Influências Observadas:**
    - Política "mulheres e crianças primeiro" fortemente aplicada
    - Normas sociais da era vitoriana refletidas nas taxas
    - Interação entre classe social e género
    - Expectativas de comportamento por género
    """)

    st.divider()

    # Implicações Históricas
    st.markdown("### 📚 Implicações Históricas")
    st.markdown("""
    **Legado do Desastre:**
    - Estabelecimento de novos protocolos de evacuação
    - Alterações nas políticas de segurança marítima
    - Impacto na perceção dos papéis de género
    - Influência nos procedimentos de emergência modernos
    """)

    st.divider()

    # Conclusões
    st.markdown("### 📊 Conclusões Principais")
    st.markdown("""
    **Observações Finais:**
    - O género foi o factor mais determinante para sobrevivência
    - A classe social amplificou as diferenças de género
    - As normas sociais tiveram papel crucial nas decisões
    - O desastre expôs e reforçou as hierarquias sociais da época
    """)

    st.divider()