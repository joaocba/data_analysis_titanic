import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise de sobrevivência por estrutura familiar"""
    st.markdown("### Análise de Sobrevivência por Estrutura Familiar")

    st.markdown("""
    Nesta secção, iremos explorar como a estrutura familiar influenciou as hipóteses de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as taxas de sobrevivência por dimensão da família, compreender os padrões de sobrevivência 
    entre diferentes composições familiares e identificar como as relações familiares afectaram a sobrevivência durante o desastre.
    """)

    # Preparar dados
    survival_by_family = pd.crosstab(data['FamilySize'], data['Survived'])
    survival_rate_family = pd.crosstab(data['FamilySize'], data['Survived'], normalize='index') * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas por Estrutura Familiar")
        stats_df = _create_family_stats_df(data)
        st.table(stats_df)

    with col2:
        st.markdown("#### Taxa de Sobrevivência")
        survival_stats_df = _create_survival_stats_df(survival_by_family, survival_rate_family)
        st.table(survival_stats_df)

    # Visualizações e Conclusões
    st.markdown("### Visualizações e Conclusões")
    _show_visualizations(data)
    _show_insights(data)


def _create_family_stats_df(data):
    """Cria DataFrame com estatísticas por estrutura familiar"""
    types = [
        ('Sozinho', data['IsAlone'] == 1),
        ('Pequena (1-3)', (data['FamilySize'] >= 1) & (data['FamilySize'] <= 3)),
        ('Média (4-6)', (data['FamilySize'] >= 4) & (data['FamilySize'] <= 6)),
        ('Numerosa (7+)', data['FamilySize'] > 6)
    ]

    stats = []
    for label, condition in types:
        group_data = data[condition]
        if len(group_data) > 0:
            stats.append({
                'Estrutura': label,
                'Total': len(group_data),
                'Idade Média': f"{group_data['Age'].mean():.1f} anos",
                'Classe Comum': f"{group_data['Pclass'].mode().iloc[0]}.ª Classe",
                'Proporção': format_percentage(len(group_data) / len(data) * 100)
            })

    return pd.DataFrame(stats)


def _create_survival_stats_df(survival_by_family, survival_rate_family):
    """Cria DataFrame com estatísticas de sobrevivência"""
    stats = []

    # Agrupar por categorias de dimensão
    size_categories = {
        'Sozinho': 0,
        'Pequena (1-3)': [1, 2, 3],
        'Média (4-6)': [4, 5, 6],
        'Numerosa (7+)': list(range(7, max(survival_by_family.index) + 1))
    }

    for category, sizes in size_categories.items():
        if isinstance(sizes, list):
            mask = survival_by_family.index.isin(sizes)
            total = survival_by_family.loc[mask].sum().sum()
            survivors = survival_by_family.loc[mask, 1].sum()
            non_survivors = total - survivors
            rate = (survivors / total * 100) if total > 0 else 0
        else:  # Caso 'Sozinho'
            total = survival_by_family.loc[sizes].sum()
            survivors = survival_by_family.loc[sizes, 1]
            non_survivors = total - survivors
            rate = (survivors / total * 100) if total > 0 else 0

        stats.append({
            'Estrutura': category,
            'Total': total,
            'Sobreviventes': survivors,
            'Não Sobreviventes': non_survivors,
            'Taxa Sobrevivência': format_percentage(rate)
        })

    return pd.DataFrame(stats)


def _show_visualizations(data):
    """Apresenta visualizações da sobrevivência por estrutura familiar"""
    st.markdown("#### Visualizações da Sobrevivência por Estrutura Familiar")
    tab1, tab2, tab3 = st.tabs(["Distribuição Geral", "Análise Detalhada", "Relações Familiares"])

    with tab1:
        _plot_survival_distribution(data)

    with tab2:
        _plot_detailed_analysis(data)

    with tab3:
        _plot_family_relations(data)


def _plot_survival_distribution(data):
    """Apresenta a distribuição geral de sobrevivência por estrutura familiar"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = get_color_palette(20)

    # Gráfico 1: Barras de sobrevivência por dimensão
    survival_by_size = pd.crosstab(data['FamilySize'], data['Survived'], normalize='index') * 100

    survival_by_size[1].plot(kind='bar', color=colors[0], ax=ax1)
    ax1.set_title('Taxa de Sobrevivência por Dimensão da Família',
                  pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Número de Familiares')
    ax1.set_ylabel('Taxa de Sobrevivência (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Adicionar rótulos nas barras
    for i, v in enumerate(survival_by_size[1]):
        ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    # Gráfico 2: Circular Sozinho vs Com Família
    alone_data = data.groupby('IsAlone')['Survived'].agg(['count', 'mean'])
    alone_data['mean'] *= 100

    sizes = alone_data['count']
    labels = ['Com Família', 'Sozinho']

    ax2.pie(sizes, labels=labels,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct * sum(sizes) / 100)})',
            colors=[colors[0], colors[2]],
            explode=(0.05, 0))
    ax2.set_title('Distribuição: Sozinho vs. Com Família',
                  pad=20, fontsize=14, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def _plot_detailed_analysis(data):
    """Apresenta análise detalhada por estrutura familiar"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular contagens e percentagens por categoria de família
    data['FamilyCategory'] = pd.cut(data['FamilySize'],
                                    bins=[-1, 0, 3, 6, np.inf],
                                    labels=['Sozinho', 'Pequena', 'Média', 'Numerosa'])

    family_survival = pd.crosstab(data['FamilyCategory'], data['Survived'])
    family_survival_pct = pd.crosstab(data['FamilyCategory'], data['Survived'], normalize='index') * 100

    # Apresentar barras empilhadas
    bottom_bars = ax.bar(range(4), family_survival[0],
                         label='Não Sobreviveu',
                         color=COLORS['negative'])
    top_bars = ax.bar(range(4), family_survival[1],
                      bottom=family_survival[0],
                      label='Sobreviveu',
                      color=COLORS['primary'])

    set_plot_style(
        ax,
        'Composição de Sobrevivência por Estrutura Familiar',
        'Categoria',
        'Número de Passageiros'
    )

    plt.xticks(range(4), ['Sozinho', 'Pequena', 'Média', 'Numerosa'])
    plt.legend(title='Estado', bbox_to_anchor=(1.0, 1.02))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar percentagens em cada barra
    for idx, rect in enumerate(bottom_bars):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                height / 2.,
                f'{family_survival_pct[0][idx]:.1f}%',
                ha='center', va='center',
                color='black', fontweight='bold')

    for idx, rect in enumerate(top_bars):
        height = rect.get_height()
        bottom = family_survival[0][idx]
        ax.text(rect.get_x() + rect.get_width() / 2.,
                bottom + height / 2.,
                f'{family_survival_pct[1][idx]:.1f}%',
                ha='center', va='center',
                color='black', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def _plot_family_relations(data):
    """Apresenta análise das relações familiares"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Criar matriz de sobrevivência por SibSp e Parch
    survival_matrix = pd.crosstab(
        [data['SibSp'], data['Parch']],
        data['Survived'],
        normalize='index'
    )[1] * 100

    survival_matrix = survival_matrix.unstack(level=0).fillna(0)

    # Criar mapa de calor
    sns.heatmap(survival_matrix,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Taxa de Sobrevivência (%)'},
                ax=ax)

    plt.title('Taxa de Sobrevivência por Tipo de Relação Familiar',
              pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Irmãos/Cônjuge (SibSp)')
    plt.ylabel('Pais/Filhos (Parch)')

    plt.tight_layout()
    st.pyplot(fig)


def _show_insights(data):
    """Apresenta as conclusões da análise de sobrevivência por estrutura familiar"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Estrutura Familiar")

    # Dividindo as secções em colunas para melhor visualização
    col1, col2 = st.columns(2)

    # Estatísticas por Grupo
    with col1:
        st.markdown("#### **Análise por Estrutura**")
        alone_data = data[data['IsAlone'] == 1]
        family_data = data[data['IsAlone'] == 0]

        st.markdown(f"""
        - **A Viajar Sozinho**:
            - Total: {len(alone_data)}
            - Sobreviventes: {len(alone_data[alone_data['Survived'] == 1])}
            - Taxa: {format_percentage(alone_data['Survived'].mean() * 100)}
        - **Com Família**:
            - Total: {len(family_data)}
            - Sobreviventes: {len(family_data[family_data['Survived'] == 1])}
            - Taxa: {format_percentage(family_data['Survived'].mean() * 100)}
        - **Dimensão Ideal**: {data.groupby('FamilySize')['Survived'].mean().idxmax()} familiares
        """)

    # Análise de Relações
    with col2:
        st.markdown("#### **Análise de Relações**")
        st.markdown("""
        - **Composição Familiar**:
            - Famílias pequenas mais comuns
            - Presença de cônjuges significativa
            - Grupos extensos mais raros
        - **Factores de Sobrevivência**:
            - Apoio mútuo importante
            - Coordenação em grupos
            - Acesso a informações
        """)

    st.divider()

    # Factores de Impacto
    st.markdown("### 👨‍👩‍👧‍👦 Factores de Impacto")
    st.markdown("""
    **Influências Observadas:**
    - Dimensão da família afecta mobilidade e coordenação
    - Relações familiares influenciam decisões
    - Estrutura familiar relacionada com a classe social
    - Dinâmicas de grupo durante a emergência
    """)

    st.divider()

    # Padrões Sociais
    st.markdown("### 🔍 Padrões Sociais")
    st.markdown("""
    **Observações Sociais:**
    - Distribuição das estruturas familiares por classe
    - Padrões de viagem da época
    - Relação entre estrutura familiar e recursos
    - Impacto das normas sociais nas famílias
    """)

    st.divider()

    # Conclusões
    st.markdown("### 📊 Conclusões Principais")
    st.markdown("""
    **Observações Finais:**
    - A estrutura familiar afectou significativamente a sobrevivência
    - Equilíbrio entre dimensão da família e taxa de sobrevivência
    - Importância do apoio familiar em situações de crise
    - Necessidade de protocolos específicos para grupos familiares
    """)

    st.divider()