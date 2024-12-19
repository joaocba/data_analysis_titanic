# correlation_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise de correlações entre variáveis"""
    st.markdown("### Análise de Correlações")

    st.markdown("""
    Nesta secção, iremos explorar as relações entre diferentes variáveis do conjunto de dados do Titanic.
    Poderá visualizar as correlações entre todas as variáveis, com foco especial nas 
    relações com a sobrevivência, além de análises detalhadas das correlações mais significativas.
    """)

    # Selecionar variáveis numéricas
    numeric_vars = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                    'Fare', 'FamilySize', 'IsAlone']
    correlation_data = data[numeric_vars]

    # Secção de Visualizações Principais
    st.markdown("### Visualizações Principais")
    tab1, tab2 = st.tabs(["Matriz de Correlação", "Correlações com Sobrevivência"])

    with tab1:
        _show_correlation_matrix(correlation_data)

    with tab2:
        _show_survival_correlations(correlation_data)

    # Análises Detalhadas
    st.markdown("### Análises Detalhadas")
    _show_detailed_analysis(data)

    # Conclusões
    st.markdown("### Conclusões")
    _show_insights()


def _show_correlation_matrix(correlation_data):
    """Apresenta matriz de correlação entre todas as variáveis"""
    st.markdown("#### Matriz de Correlação Geral")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Calcular matriz de correlação
        correlation_matrix = correlation_data.corr()

        # Criar mapa de calor
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap='RdYlBu_r',
                    center=0,
                    fmt='.2f',
                    square=True)

        plt.title('Matriz de Correlação entre Variáveis',
                  pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        **Interpretação das Correlações:**
        - **1,0 a 0,7**: Correlação forte positiva
        - **0,7 a 0,3**: Correlação moderada positiva
        - **0,3 a -0,3**: Correlação fraca
        - **-0,3 a -0,7**: Correlação moderada negativa
        - **-0,7 a -1,0**: Correlação forte negativa

        **Observações:**
        - Cores mais intensas indicam correlações mais fortes
        - Azul indica correlação positiva
        - Vermelho indica correlação negativa
        """)


def _show_survival_correlations(correlation_data):
    """Apresenta correlações específicas com sobrevivência"""
    st.markdown("#### Correlações com Sobrevivência")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Calcular correlações com sobrevivência
        survival_corr = correlation_data.corr()['Survived'].sort_values(ascending=False)

        # Criar DataFrame para exibição
        corr_df = pd.DataFrame({
            'Variável': [
                'Sobrevivência',
                'Sexo (F=1, M=0)',
                'Classe',
                'Tarifa',
                'A Viajar Sozinho',
                'Pais/Filhos',
                'Irmãos/Cônjuge',
                'Idade',
                'Dimensão da Família'
            ],
            'Correlação': survival_corr.values,
            'Força': [
                'Auto-correlação',
                'Forte positiva',
                'Forte negativa',
                'Moderada positiva',
                'Fraca negativa',
                'Fraca positiva',
                'Fraca negativa',
                'Fraca negativa',
                'Muito fraca'
            ]
        })
        st.table(corr_df)

    with col2:
        st.markdown("""
        **Principais Factores:**

        🔵 **Impacto Positivo na Sobrevivência:**
        - Ser do sexo feminino
        - Pagar tarifa mais elevada
        - Viajar com pais/filhos

        🔴 **Impacto Negativo na Sobrevivência:**
        - Classe mais baixa
        - Viajar sozinho
        - Idade mais avançada
        """)


def _show_detailed_analysis(data):
    """Apresenta análise detalhada das correlações mais importantes"""
    tab1, tab2, tab3 = st.tabs([
        "Sexo vs. Classe",
        "Idade vs. Tarifa",
        "Família vs. Tarifa"
    ])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            _plot_sex_class_correlation(data)
        with col2:
            st.markdown("""
            **Observações:**
            - Maior proporção de mulheres na 1.ª classe
            - 3.ª classe predominantemente masculina
            - Distribuição mais equilibrada na 2.ª classe
            """)

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            _plot_age_fare_correlation(data)
        with col2:
            st.markdown("""
            **Observações:**
            - Tarifas mais elevadas para passageiros mais velhos
            - Maior variação de tarifas na 1.ª classe
            - Padrão de sobrevivência mais claro por faixa de preço
            """)

    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            _plot_family_fare_correlation(data)
        with col2:
            st.markdown("""
            **Observações:**
            - Famílias maiores tendem a pagar tarifas mais elevadas
            - Possível relação com cabinas maiores
            - Descontos para grupos podem influenciar
            """)


def _plot_sex_class_correlation(data):
    """Apresenta correlação entre sexo e classe"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Verificar e limpar dados, removendo valores nulos
    data_cleaned = data[['Pclass', 'Sex']].dropna()

    # Calcular percentagens por classe e sexo
    class_sex_dist = pd.crosstab(data_cleaned['Pclass'], data_cleaned['Sex'], normalize='index') * 100

    # Apresentar gráfico de barras com as percentagens
    class_sex_dist.plot(kind='bar',
                        ax=ax,
                        color=[COLORS['negative'], COLORS['primary']],
                        width=0.8,
                        edgecolor='white')

    # Ajustar título e rótulos
    ax.set_title('Distribuição de Género por Classe', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classe', fontsize=12)
    ax.set_ylabel('Percentagem (%)', fontsize=12)
    ax.set_xticklabels(['1.ª Classe', '2.ª Classe', '3.ª Classe'], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(['Masculino', 'Feminino'], fontsize=10)

    # Adicionar rótulos nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center', color='black', fontweight='bold', fontsize=10)

    # Ajustar estilo do gráfico
    ax.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')

    # Apresentar gráfico no Streamlit
    st.pyplot(fig)


def _plot_age_fare_correlation(data):
    """Apresenta correlação entre idade e tarifa"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Certificar-se de que 'Survived' tem valores consistentes
    data['Survived'] = data['Survived'].map({0: 'Não Sobreviveu', 1: 'Sobreviveu'})

    # Usar uma paleta de cores que combine com a legenda
    sns.scatterplot(data=data,
                    x='Age',
                    y='Fare',
                    hue='Survived',
                    palette=[COLORS['negative'], COLORS['primary']],
                    alpha=0.6)

    set_plot_style(
        ax,
        'Relação entre Idade e Tarifa',
        'Idade',
        'Tarifa (£)'
    )
    plt.legend(title='Sobrevivência')

    st.pyplot(fig)


def _plot_family_fare_correlation(data):
    """Apresenta correlação entre dimensão da família e tarifa"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Verificar a relação entre FamilySize e Fare
    family_fare = data.groupby('FamilySize')['Fare'].mean().reset_index()

    # Ajustar a paleta de cores para um gradiente mais atraente
    palette = sns.color_palette("viridis", len(family_fare))

    # Apresentar a relação média entre dimensão da família e tarifa
    sns.barplot(data=family_fare,
                x='FamilySize',
                y='Fare',
                palette=palette,
                ax=ax)

    # Adicionar rótulos nas barras para mostrar os valores da tarifa média
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='black',
                    fontweight='bold',
                    xytext=(0, 5),
                    textcoords='offset points')

    # Melhorar o título, rótulos e estilo
    ax.set_title('Média das Tarifas por Dimensão da Família', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Familiares', fontsize=12)
    ax.set_ylabel('Tarifa (£)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')

    # Apresentar gráfico no Streamlit
    st.pyplot(fig)


def _show_insights():
    """Apresenta conclusões sobre as correlações"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Factores Principais de Sobrevivência")
        st.markdown("""
        **1. Características Demográficas:**
        - **Sexo**: Principal factor de sobrevivência
        - **Classe Social**: Segundo factor mais importante
        - **Idade**: Impacto moderado, principalmente com outros factores

        **2. Factores Socioeconómicos:**
        - Forte relação entre tarifa e sobrevivência
        - Qualidade das acomodações influencia hipóteses
        - Padrões sociais reflectidos nas correlações
        """)

    with col2:
        st.markdown("#### Padrões Sociais e Familiares")
        st.markdown("""
        **1. Estrutura Familiar:**
        - Famílias maiores em classes superiores
        - Apoio familiar como factor de sobrevivência
        - Relação entre dimensão da família e tarifa

        **2. Aspectos Socioeconómicos:**
        - Distribuição demográfica por classe
        - Padrões de viagem por grupo social
        - Relação entre idade e classe social
        """)

    st.divider()

    st.markdown("### 📝 Conclusões Principais")
    st.markdown("""
    1. **Hierarquia de Factores:**
       - Género e classe social como determinantes principais
       - Factores familiares com influência moderada
       - Idade com impacto contextual

    2. **Interacções Complexas:**
       - Múltiplas variáveis interagindo simultaneamente
       - Padrões sociais reflectidos nos dados
       - Importância da análise multivariada
    """)

    st.divider()