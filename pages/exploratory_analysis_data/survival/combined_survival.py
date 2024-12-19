import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise combinada de factores de sobrevivência"""
    st.markdown("### Análise Combinada de Factores de Sobrevivência")

    st.markdown("""
    Nesta secção, iremos explorar como diferentes factores interagiram para influenciar as hipóteses de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar a importância relativa de cada característica, compreender como estas se relacionam entre si 
    e identificar os padrões complexos que determinaram a sobrevivência durante o desastre.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Correlação com Sobrevivência")
        correlation_df = _create_correlation_df(data)
        st.table(correlation_df)

    with col2:
        st.markdown("#### Importância das Características")
        importance_df = _create_importance_df(data)
        st.table(importance_df)

    # Visualizações e Conclusões
    st.markdown("### Visualizações e Conclusões")
    _show_visualizations(data)
    _show_insights(data)


def _create_correlation_df(data):
    """Cria DataFrame com correlações"""
    numeric_vars = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                    'Fare', 'FamilySize', 'IsAlone']

    correlations = data[numeric_vars].corr()['Survived'].sort_values(ascending=False)

    return pd.DataFrame({
        'Variável': correlations.index[1:],  # Excluir autocorrelação
        'Correlação': [f"{x:.3f}" for x in correlations.values[1:]],
        'Impacto': ['Forte' if abs(x) > 0.3 else 'Moderado' if abs(x) > 0.1 else 'Fraco'
                    for x in correlations.values[1:]]
    })


def _create_importance_df(data):
    """Cria DataFrame com importância das características"""
    # Calcular impactos
    gender_impact = abs(data[data['Sex'] == 1]['Survived'].mean() -
                        data[data['Sex'] == 0]['Survived'].mean()) * 100

    class_impact = abs(data[data['Pclass'] == 1]['Survived'].mean() -
                       data[data['Pclass'] == 3]['Survived'].mean()) * 100

    age_impact = abs(data[data['Age'] <= 17]['Survived'].mean() -
                     data[data['Age'] > 17]['Survived'].mean()) * 100

    family_impact = abs(data[data['IsAlone'] == 1]['Survived'].mean() -
                        data[data['IsAlone'] == 0]['Survived'].mean()) * 100

    return pd.DataFrame({
        'Característica': ['Género', 'Classe', 'Idade', 'Estrutura Familiar'],
        'Impacto': [
            format_percentage(gender_impact),
            format_percentage(class_impact),
            format_percentage(age_impact),
            format_percentage(family_impact)
        ]
    })


def _show_visualizations(data):
    """Apresenta visualizações das interacções"""
    st.markdown("#### Visualizações das Interacções")
    tab1, tab2, tab3 = st.tabs(["Importância Relativa", "Género e Classe", "Idade e Classe"])

    with tab1:
        _plot_feature_importance(data)

    with tab2:
        _plot_gender_class_interaction(data)

    with tab3:
        _plot_age_class_interaction(data)


def _plot_feature_importance(data):
    """Apresenta a importância relativa das características"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular impactos
    impacts = {
        'Género': abs(data[data['Sex'] == 1]['Survived'].mean() -
                      data[data['Sex'] == 0]['Survived'].mean()) * 100,
        'Classe': abs(data[data['Pclass'] == 1]['Survived'].mean() -
                      data[data['Pclass'] == 3]['Survived'].mean()) * 100,
        'Idade': abs(data[data['Age'] <= 17]['Survived'].mean() -
                     data[data['Age'] > 17]['Survived'].mean()) * 100,
        'Estrutura\nFamiliar': abs(data[data['IsAlone'] == 1]['Survived'].mean() -
                                   data[data['IsAlone'] == 0]['Survived'].mean()) * 100
    }

    # Ordenar por impacto
    impacts = dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))

    bars = plt.bar(impacts.keys(), impacts.values(), color=COLORS['primary'])

    set_plot_style(
        ax,
        'Impacto Relativo das Características na Sobrevivência',
        'Característica',
        'Diferença na Taxa de Sobrevivência (%)'
    )

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)


def _plot_gender_class_interaction(data):
    """Apresenta a interacção entre género e classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular taxas de sobrevivência
    survival_rates = pd.crosstab([data['Sex'], data['Pclass']],
                                 data['Survived'],
                                 normalize='index') * 100

    # Preparar dados para o gráfico
    x = np.arange(3)
    width = 0.35

    men_rates = [survival_rates.loc[(0, i + 1), 1] for i in range(3)]
    women_rates = [survival_rates.loc[(1, i + 1), 1] for i in range(3)]

    # Criar barras
    bars1 = ax.bar(x - width / 2, men_rates, width,
                   label='Masculino', color=COLORS['negative'])
    bars2 = ax.bar(x + width / 2, women_rates, width,
                   label='Feminino', color=COLORS['primary'])

    set_plot_style(
        ax,
        'Interacção entre Género e Classe na Sobrevivência',
        'Classe',
        'Taxa de Sobrevivência (%)'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(['1.ª Classe', '2.ª Classe', '3.ª Classe'])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height / 2.,
                    f'{height:.1f}%',
                    ha='center', va='center',
                    color='black', fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    st.pyplot(fig)


def _plot_age_class_interaction(data):
    """Apresenta a interacção entre idade e classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Criar grupos de idade
    data['AgeGroup'] = pd.cut(data['Age'],
                              bins=[0, 17, 50, 100],
                              labels=['Criança', 'Adulto', 'Idoso'])

    # Calcular taxas de sobrevivência
    survival_rates = pd.crosstab([data['AgeGroup'], data['Pclass']],
                                 data['Survived'],
                                 normalize='index') * 100

    # Preparar dados para o gráfico
    survival_rates = survival_rates[1].unstack()  # Obter apenas taxa de sobrevivência

    # Criar mapa de calor
    sns.heatmap(survival_rates,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Taxa de Sobrevivência (%)'},
                ax=ax)

    plt.title('Taxa de Sobrevivência por Idade e Classe',
              pad=20, fontsize=14, fontweight='bold')

    # Atualizar rótulos
    plt.xlabel('Classe')
    plt.ylabel('Grupo de Idade')

    plt.tight_layout()
    st.pyplot(fig)


def _show_insights(data):
    """Apresenta as conclusões da análise combinada"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Factores Combinados")

    # Dividindo as secções em colunas para melhor visualização
    col1, col2 = st.columns(2)

    # Hierarquia de Factores
    with col1:
        st.markdown("#### **Hierarquia de Factores**")
        st.markdown("""
        - **Género**:
            - Factor mais determinante
            - Política "mulheres primeiro"
            - Forte influência independente
        - **Classe Social**:
            - Segundo factor mais importante
            - Forte interacção com género
            - Impacto na acessibilidade
        - **Idade**:
            - Importância moderada
            - Protecção especial a crianças
            - Interacção com género e classe
        """)

    # Padrões de Interacção
    with col2:
        st.markdown("#### **Padrões de Interacção**")
        st.markdown("""
        - **Género + Classe**:
            - Mulheres 1.ª classe: maior taxa
            - Homens 3.ª classe: menor taxa
            - Diferenças mais pronunciadas
        - **Idade + Classe**:
            - Crianças: protecção em todas classes
            - Adultos: grande variação
            - Idosos: mais vulneráveis
        """)

    st.divider()

    # Factores Socioeconómicos
    st.markdown("### 💰 Factores Socioeconómicos")
    st.markdown("""
    **Impactos Observados:**
    - Classe social como multiplicador de hipóteses
    - Acesso diferenciado a recursos de salvamento
    - Interacção entre estatuto social e prioridades
    - Reflexo da estrutura social da época
    """)

    st.divider()

    # Implicações
    st.markdown("### 🔍 Implicações")
    st.markdown("""
    **Conclusões Importantes:**
    - Múltiplos factores determinaram a sobrevivência
    - Interacções complexas entre características
    - Padrões claros de privilégio e discriminação
    - Importância de análise multidimensional
    """)

    st.divider()

    # Lições Históricas
    st.markdown("### 📚 Lições Históricas")
    st.markdown("""
    **Aprendizagens:**
    - Necessidade de protocolos claros de emergência
    - Importância de considerar grupos vulneráveis
    - Influência de factores sociais em crises
    - Valor da análise integrada de factores
    """)

    st.divider()