import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise de sobrevivência por porto de embarque"""
    st.markdown("### Análise de Sobrevivência por Porto de Embarque")

    st.markdown("""
    Nesta secção, iremos explorar como o porto de embarque influenciou as hipóteses de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as taxas de sobrevivência por porto, compreender os padrões socioeconómicos 
    associados a cada local de embarque e identificar como estas características afectaram a sobrevivência.
    """)

    # Criar colunas dummies para os valores de Embarked
    data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas por Porto")
        stats_df = _create_port_stats_df(data)
        st.table(stats_df)

    with col2:
        st.markdown("#### Taxa de Sobrevivência")
        survival_stats_df = _create_survival_stats_df(data)
        st.table(survival_stats_df)

    # Visualizações e Conclusões
    st.markdown("### Visualizações e Conclusões")
    _show_visualizations(data)
    _show_insights(data)


def _create_port_stats_df(data):
    """Cria DataFrame com estatísticas por porto"""
    ports = {
        'S': ('Embarked_S', 'Southampton'),
        'C': ('Embarked_C', 'Cherburgo'),
        'Q': ('Embarked_Q', 'Queenstown')
    }

    stats = []
    for code, (col, name) in ports.items():
        port_data = data[data[col] == 1]
        stats.append({
            'Porto': name,
            'Total': len(port_data),
            'Idade Média': f"{port_data['Age'].mean():.1f} anos",
            'Tarifa Média': f"£{port_data['Fare'].mean():.2f}",
            'Proporção': format_percentage(len(port_data) / len(data) * 100)
        })

    return pd.DataFrame(stats)


def _create_survival_stats_df(data):
    """Cria DataFrame com estatísticas de sobrevivência"""
    ports = {
        'S': ('Embarked_S', 'Southampton'),
        'C': ('Embarked_C', 'Cherburgo'),
        'Q': ('Embarked_Q', 'Queenstown')
    }

    stats = []
    for code, (col, name) in ports.items():
        port_data = data[data[col] == 1]
        survivors = len(port_data[port_data['Survived'] == 1])
        non_survivors = len(port_data[port_data['Survived'] == 0])
        total = survivors + non_survivors

        stats.append({
            'Porto': name,
            'Total': total,
            'Sobreviventes': survivors,
            'Não Sobreviventes': non_survivors,
            'Taxa Sobrevivência': format_percentage((survivors / total * 100) if total > 0 else 0)
        })

    return pd.DataFrame(stats)


def _show_visualizations(data):
    """Apresenta visualizações da sobrevivência por porto"""
    st.markdown("#### Visualizações da Sobrevivência por Porto")
    tab1, tab2, tab3 = st.tabs(["Distribuição Geral", "Por Classe", "Análise Detalhada"])

    with tab1:
        _plot_survival_distribution(data)

    with tab2:
        _plot_class_distribution(data)

    with tab3:
        _plot_detailed_analysis(data)


def _plot_survival_distribution(data):
    """Apresenta a distribuição geral de sobrevivência por porto"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular contagens e percentagens
    ports = {
        'Southampton': data['Embarked_S'] == 1,
        'Cherburgo': data['Embarked_C'] == 1,
        'Queenstown': data['Embarked_Q'] == 1
    }

    survival_data = []
    positions = np.arange(len(ports))
    width = 0.35

    for survived in [0, 1]:
        counts = []
        for mask in ports.values():
            port_data = data[mask]
            count = len(port_data[port_data['Survived'] == survived])
            counts.append(count)

        if survived == 0:
            bottom_bars = ax.bar(positions - width / 2, counts, width,
                                 label='Não Sobreviveu', color=COLORS['negative'])
        else:
            top_bars = ax.bar(positions + width / 2, counts, width,
                              label='Sobreviveu', color=COLORS['primary'])

    set_plot_style(
        ax,
        'Distribuição de Sobrevivência por Porto',
        'Porto',
        'Número de Passageiros'
    )

    plt.xticks(positions, ports.keys())
    plt.legend(title='Estado', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adicionar rótulos nas barras
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height / 2.,
                    f'{int(height)}',
                    ha='center', va='center', color='black', fontweight='bold')

    add_labels(bottom_bars)
    add_labels(top_bars)

    plt.tight_layout()
    st.pyplot(fig)


def _plot_class_distribution(data):
    """Apresenta a distribuição de sobrevivência por porto e classe"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular taxas de sobrevivência por porto e classe
    survival_rates = pd.DataFrame(index=['1.ª Classe', '2.ª Classe', '3.ª Classe'])

    ports = {
        'S': ('Embarked_S', 'Southampton'),
        'C': ('Embarked_C', 'Cherburgo'),
        'Q': ('Embarked_Q', 'Queenstown')
    }

    for code, (col, name) in ports.items():
        rates = []
        for pclass in [1, 2, 3]:
            mask = (data[col] == 1) & (data['Pclass'] == pclass)
            rate = (data[mask]['Survived'].mean() * 100) if mask.any() else 0
            rates.append(rate)
        survival_rates[name] = rates

    # Criar mapa de calor
    sns.heatmap(survival_rates,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Taxa de Sobrevivência (%)'},
                ax=ax)

    plt.title('Taxa de Sobrevivência por Porto e Classe',
              pad=20, fontsize=14, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)


def _plot_detailed_analysis(data):
    """Apresenta análise detalhada por porto"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Preparar dados para análise de tarifa média por porto e sobrevivência
    ports = {
        'Southampton': data['Embarked_S'] == 1,
        'Cherburgo': data['Embarked_C'] == 1,
        'Queenstown': data['Embarked_Q'] == 1
    }

    avg_fares = []
    survival_rates = []
    port_names = []

    for name, mask in ports.items():
        port_data = data[mask]
        avg_fares.append(port_data['Fare'].mean())
        survival_rates.append(port_data['Survived'].mean() * 100)
        port_names.append(name)

    # Criar gráfico de dispersão
    scatter = plt.scatter(avg_fares, survival_rates,
                          c=range(len(ports)),
                          cmap='viridis',
                          s=200)

    # Adicionar rótulos
    for i, txt in enumerate(port_names):
        plt.annotate(txt, (avg_fares[i], survival_rates[i]),
                     xytext=(5, 5), textcoords='offset points')

    set_plot_style(
        ax,
        'Relação entre Tarifa Média e Sobrevivência por Porto',
        'Tarifa Média (£)',
        'Taxa de Sobrevivência (%)'
    )

    plt.tight_layout()
    st.pyplot(fig)


def _show_insights(data):
    """Apresenta as conclusões da análise de sobrevivência por porto"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Porto de Embarque")

    # Dividindo as secções em colunas para melhor visualização
    col1, col2 = st.columns(2)

    # Estatísticas por Porto
    with col1:
        st.markdown("#### **Análise por Porto**")
        ports = {
            'Southampton': data[data['Embarked_S'] == 1],
            'Cherburgo': data[data['Embarked_C'] == 1],
            'Queenstown': data[data['Embarked_Q'] == 1]
        }

        for name, port_data in ports.items():
            survivors = len(port_data[port_data['Survived'] == 1])
            total = len(port_data)
            first_class = len(port_data[port_data['Pclass'] == 1])
            avg_fare = port_data['Fare'].mean()

            st.markdown(f"""
            - **{name}**:
                - Passageiros: {total}
                - Sobreviventes: {survivors}
                - Taxa: {format_percentage((survivors / total * 100) if total > 0 else 0)}
                - Primeira Classe: {format_percentage((first_class / total * 100) if total > 0 else 0)}
                - Tarifa Média: £{avg_fare:.2f}
            """)

    # Padrões Socioeconómicos
    with col2:
        st.markdown("#### **Padrões Socioeconómicos**")
        st.markdown("""
        - **Southampton**:
            - Principal porto britânico
            - Distribuição equilibrada de classes
            - Rota principal para a América
        - **Cherburgo**:
            - Mais passageiros de 1.ª classe
            - Tarifas médias mais elevadas
            - Perfil mais abastado
        - **Queenstown**:
            - Predominância de 3.ª classe
            - Emigração irlandesa
            - Tarifas mais reduzidas
        """)

    st.divider()

    # Factores Geográficos
    st.markdown("### 🌍 Factores Geográficos")
    st.markdown("""
    **Características Regionais:**
    - Southampton: principal porto de partida
    - Cherburgo: porto francês de luxo
    - Queenstown: porto de emigração irlandesa
    - Diferentes perfis socioeconómicos por região
    """)

    st.divider()

    # Padrões Migratórios
    st.markdown("### 🚢 Padrões Migratórios")
    st.markdown("""
    **Tendências da Época:**
    - Rotas estabelecidas de emigração europeia
    - Perfis distintos de passageiros por porto
    - Relação com condições económicas locais
    - Influência nas taxas de sobrevivência
    """)

    st.divider()

    # Conclusões
    st.markdown("### 📊 Conclusões Principais")
    st.markdown("""
    **Observações Finais:**
    - Porto de embarque reflecte perfil socioeconómico
    - Relação directa com taxas de sobrevivência
    - Padrões históricos de emigração evidentes
    - Importância dos portos na era dos transatlânticos
    """)

    st.divider()