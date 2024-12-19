import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise de sobrevivência por classe"""
    st.markdown("### Análise de Sobrevivência por Classe")

    st.markdown("""
    Nesta secção, iremos explorar como a classe social influenciou as hipóteses de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as taxas de sobrevivência por classe, compreender os padrões socioeconómicos 
    e identificar como a hierarquia social da época afectou os resultados durante o desastre.
    """)

    # Preparar dados
    survival_by_class = pd.crosstab(data['Pclass'], data['Survived'])
    survival_rate_class = pd.crosstab(data['Pclass'], data['Survived'], normalize='index') * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas por Classe")
        stats_df = _create_class_stats_df(data)
        st.table(stats_df)

    with col2:
        st.markdown("#### Taxa de Sobrevivência")
        survival_stats_df = _create_survival_stats_df(survival_by_class, survival_rate_class)
        st.table(survival_stats_df)

    # Visualizações e Conclusões
    st.markdown("### Visualizações e Conclusões")
    _show_visualizations(data, survival_by_class)
    _show_insights(data)


def _create_class_stats_df(data):
    """Cria DataFrame com estatísticas por classe"""
    stats = []

    for pclass in [1, 2, 3]:
        class_data = data[data['Pclass'] == pclass]
        stats.append({
            'Classe': f'{pclass}.ª Classe',
            'Total': len(class_data),
            'Idade Média': f"{class_data['Age'].mean():.1f} anos",
            'Tarifa Média': f"£{class_data['Fare'].mean():.2f}",
            'Proporção': format_percentage(len(class_data) / len(data) * 100)
        })

    return pd.DataFrame(stats)


def _create_survival_stats_df(survival_by_class, survival_rate_class):
    """Cria DataFrame com estatísticas de sobrevivência"""
    return pd.DataFrame({
        'Classe': ['1.ª Classe', '2.ª Classe', '3.ª Classe'],
        'Total': [survival_by_class.loc[i].sum() for i in [1, 2, 3]],
        'Sobreviventes': [survival_by_class.loc[i, 1] for i in [1, 2, 3]],
        'Não Sobreviventes': [survival_by_class.loc[i, 0] for i in [1, 2, 3]],
        'Taxa Sobrevivência': [format_percentage(survival_rate_class.loc[i, 1]) for i in [1, 2, 3]]
    })


def _show_visualizations(data, survival_by_class):
    """Apresenta visualizações da sobrevivência por classe"""
    st.markdown("#### Visualizações da Sobrevivência por Classe")
    tab1, tab2, tab3 = st.tabs(["Distribuição Geral", "Por Faixa Etária", "Análise Detalhada"])

    with tab1:
        _plot_survival_distribution(data, survival_by_class)

    with tab2:
        _plot_age_distribution(data)

    with tab3:
        _plot_detailed_analysis(data)


def _plot_survival_distribution(data, survival_by_class):
    """Apresenta a distribuição geral de sobrevivência por classe"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = get_color_palette(20)

    x = np.arange(3)
    width = 0.35

    # Criar as barras
    non_survivors = plt.bar(x - width / 2, [survival_by_class.loc[i, 0] for i in [1, 2, 3]],
                            width, label='Não Sobreviveu', color=colors[3])
    survivors = plt.bar(x + width / 2, [survival_by_class.loc[i, 1] for i in [1, 2, 3]],
                        width, label='Sobreviveu', color=colors[0])

    set_plot_style(
        ax,
        'Distribuição de Sobrevivência por Classe',
        'Classe',
        'Número de Passageiros'
    )

    plt.xticks(x, ['1.ª Classe', '2.ª Classe', '3.ª Classe'])
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
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


def _plot_age_distribution(data):
    """Apresenta a distribuição de idade por classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Criar violin plot
    sns.violinplot(data=data, x='Pclass', y='Age', hue='Survived',
                   palette=[COLORS['negative'], COLORS['primary']])

    set_plot_style(
        ax,
        'Distribuição de Idade por Classe e Sobrevivência',
        'Classe',
        'Idade (anos)'
    )

    # Atualizar rótulos do eixo x
    plt.xticks([0, 1, 2], ['1.ª Classe', '2.ª Classe', '3.ª Classe'])
    plt.legend(title='Estado', labels=['Não Sobreviveu', 'Sobreviveu'])
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)


def _plot_detailed_analysis(data):
    """Apresenta análise detalhada por classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular contagens e percentagens
    class_survival = pd.crosstab(data['Pclass'], data['Survived'])
    class_survival_pct = pd.crosstab(data['Pclass'], data['Survived'], normalize='index') * 100

    # Converter para arrays
    non_survivors = class_survival[0].values
    survivors = class_survival[1].values
    non_survivors_pct = class_survival_pct[0].values
    survivors_pct = class_survival_pct[1].values

    # Apresentar barras empilhadas
    bottom_bars = ax.bar([0, 1, 2], non_survivors,
                         label='Não Sobreviveu',
                         color=COLORS['negative'])
    top_bars = ax.bar([0, 1, 2], survivors,
                      bottom=non_survivors,
                      label='Sobreviveu',
                      color=COLORS['primary'])

    set_plot_style(
        ax,
        'Composição de Sobrevivência por Classe',
        'Classe',
        'Número de Passageiros'
    )

    plt.xticks([0, 1, 2], ['1.ª Classe', '2.ª Classe', '3.ª Classe'])
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
    """Apresenta as conclusões da análise de sobrevivência por classe"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Classe e Sobrevivência")

    # Dividindo as secções em colunas para melhor visualização
    col1, col2 = st.columns(2)

    # Estatísticas por Classe
    with col1:
        st.markdown("#### **Análise por Classe**")
        class_stats = {}
        for pclass in [1, 2, 3]:
            class_data = data[data['Pclass'] == pclass]
            survivors = len(class_data[class_data['Survived'] == 1])
            total = len(class_data)
            class_stats[pclass] = {
                'total': total,
                'survivors': survivors,
                'rate': (survivors / total * 100),
                'fare': class_data['Fare'].mean()
            }

        st.markdown(f"""
        - **1.ª Classe**:
            - Total: {class_stats[1]['total']}
            - Sobreviventes: {class_stats[1]['survivors']}
            - Taxa: {format_percentage(class_stats[1]['rate'])}
            - Tarifa média: £{class_stats[1]['fare']:.2f}
        - **2.ª Classe**:
            - Total: {class_stats[2]['total']}
            - Sobreviventes: {class_stats[2]['survivors']}
            - Taxa: {format_percentage(class_stats[2]['rate'])}
            - Tarifa média: £{class_stats[2]['fare']:.2f}
        - **3.ª Classe**:
            - Total: {class_stats[3]['total']}
            - Sobreviventes: {class_stats[3]['survivors']}
            - Taxa: {format_percentage(class_stats[3]['rate'])}
            - Tarifa média: £{class_stats[3]['fare']:.2f}
        """)

    # Padrões Observados
    with col2:
        st.markdown("#### **Padrões Observados**")
        st.markdown("""
        - **Hierarquia Social**:
            - Clara correlação classe-sobrevivência
            - Privilégio reflectido nas taxas
            - Acesso diferenciado aos recursos
        - **Factores Físicos**:
            - Localização das cabinas
            - Proximidade aos botes
            - Acesso à informação
        - **Aspectos Económicos**:
            - Relação tarifa-sobrevivência
            - Diferenças de tratamento
            - Impacto do poder aquisitivo
        """)

    st.divider()

    # Conclusões Principais
    st.markdown("### 📊 Conclusões Principais")
    st.markdown("""
    **Observações Centrais:**
    - Forte correlação entre classe social e sobrevivência
    - Primeira classe com hipóteses significativamente maiores
    - Terceira classe enfrentou maiores dificuldades
    - Padrão consistente em todas as faixas etárias
    """)

    st.divider()

    # Implicações Históricas
    st.markdown("### 🔍 Implicações Históricas")
    st.markdown("""
    **Relevância Histórica:**
    - Reflexo da estratificação social da época
    - Influência nas políticas de segurança marítima
    - Questionamento sobre igualdade em emergências
    - Debate sobre privilégios em situações críticas
    """)

    st.divider()

    # Legado
    st.markdown("### 📚 Legado")
    st.markdown("""
    **Impacto Duradouro:**
    - Alterações nas regulamentações marítimas
    - Novos protocolos de evacuação
    - Maior consciência sobre desigualdade social
    - Influência em políticas de emergência modernas
    """)

    st.divider()