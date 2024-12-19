import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage


def show(data):
    """Análise geral de sobrevivência"""
    st.markdown("### Análise Geral de Sobrevivência")

    st.markdown("""
    Nesta secção, iremos explorar os padrões gerais de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as estatísticas principais, a distribuição temporal dos acontecimentos, 
    bem como obter informações sobre os factores que influenciaram a sobrevivência durante o desastre.
    """)

    # Calcular estatísticas básicas
    survivors = data['Survived'].value_counts()
    survival_rate = (survivors[1] / len(data)) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas da Sobrevivência")
        stats_df = _create_survival_stats_df(data, survivors, survival_rate)
        st.table(stats_df)

    with col2:
        st.markdown("#### Distribuição de Sobrevivência")
        _plot_survival_distribution(data, survivors)

    # Análise Temporal
    st.markdown("### Análise Temporal do Desastre")
    _show_timeline_analysis()

    # Visualizações Detalhadas
    st.markdown("### Visualizações e Análises Detalhadas")
    _show_visualizations(data)
    _show_insights(data)


def _create_survival_stats_df(data, survivors, survival_rate):
    """Cria DataFrame com estatísticas detalhadas de sobrevivência"""
    return pd.DataFrame({
        'Estatística': [
            'Total de passageiros',
            'Sobreviventes',
            'Não sobreviventes',
            'Taxa de sobrevivência',
            'Taxa de mortalidade',
            'Rácio mortos/sobreviventes',
            'Idade média sobreviventes',
            'Idade média não sobreviventes',
            'Tarifa média sobreviventes',
            'Tarifa média não sobreviventes'
        ],
        'Valor': [
            f"{len(data)}",
            f"{survivors[1]}",
            f"{survivors[0]}",
            f"{format_percentage(survival_rate)}",
            f"{format_percentage(100 - survival_rate)}",
            f"{(survivors[0] / survivors[1]):.2f}",
            f"{data[data['Survived'] == 1]['Age'].mean():.1f} anos",
            f"{data[data['Survived'] == 0]['Age'].mean():.1f} anos",
            f"£{data[data['Survived'] == 1]['Fare'].mean():.2f}",
            f"£{data[data['Survived'] == 0]['Fare'].mean():.2f}"
        ]
    })


def _show_timeline_analysis():
    """Mostra análise temporal detalhada do desastre"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cronologia do Desastre")
        timeline = pd.DataFrame({
            'Hora': ['23:40', '00:00', '00:15', '00:45', '01:15', '01:45', '02:05', '02:20'],
            'Acontecimento': [
                'Colisão com icebergue',
                'Avaliação de danos',
                'Preparação dos botes',
                'Primeiros botes lançados',
                'Evacuação principal',
                'Últimos botes regulares',
                'Botes desmontáveis',
                'Naufrágio completo'
            ],
            'Evacuados': ['0', '~20', '~35', '~130', '~270', '~315', '~340', '340'],
            'Fase': [
                'Inicial',
                'Avaliação',
                'Preparação',
                'Evacuação Inicial',
                'Evacuação Principal',
                'Evacuação Final',
                'Fase Crítica',
                'Naufrágio'
            ]
        })
        st.table(timeline)

    with col2:
        st.markdown("#### Capacidade de Salvamento")
        capacity_df = pd.DataFrame({
            'Recurso': [
                'Botes regulares',
                'Botes desmontáveis',
                'Capacidade total',
                'Capacidade utilizada',
                'Eficiência de utilização',
                'Défice de capacidade',
                'Impacto na sobrevivência',
                'Tempo médio de evacuação'
            ],
            'Valor': [
                '16 unidades (1048 pessoas)',
                '4 unidades (130 pessoas)',
                '1178 pessoas',
                '340 pessoas',
                '28,9% da capacidade',
                '838 lugares não utilizados',
                'Decisivo para 61,6% das mortes',
                '~8 minutos por bote'
            ]
        })
        st.table(capacity_df)


def _show_visualizations(data):
    """Mostra visualizações detalhadas da sobrevivência"""
    st.markdown("#### Visualizações da Sobrevivência")
    tab1, tab2, tab3 = st.tabs([
        "Evolução Temporal",
        "Análise por Classe e Sobrevivência",
        "Distribuição de Idade e Tarifa"
    ])

    with tab1:
        _plot_evacuation_timeline()

    with tab2:
        _plot_survival_factors(data)

    with tab3:
        _plot_age_fare_distribution(data)


def _plot_survival_distribution(data, survivors):
    """Apresenta a distribuição geral de sobrevivência"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_color_palette(20)

    plt.pie(survivors.values,
            labels=['Não Sobreviveu', 'Sobreviveu'],
            autopct=lambda pct: f'{format_percentage(pct)}\n({int(pct * len(data) / 100)})',
            colors=[COLORS['negative'], COLORS['primary']],
            explode=(0.05, 0.05),
            startangle=90)

    plt.title('Taxa de Sobrevivência dos Passageiros',
              pad=20, fontsize=14, fontweight='bold')
    st.pyplot(fig)


def _plot_evacuation_timeline():
    """Apresenta a evolução temporal detalhada da evacuação"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Dados temporais
    times = ['23:40', '00:00', '00:15', '00:45', '01:15', '01:45', '02:05', '02:20']
    evacuated = [0, 20, 35, 130, 270, 315, 340, 340]
    rate = [0, 20, 15, 95, 140, 45, 25, 0]  # Taxa de evacuação por intervalo

    # Gráfico de evolução
    ax1.plot(times, evacuated, 'o-', linewidth=2, markersize=8, color=COLORS['primary'])
    ax1.fill_between(times, evacuated, alpha=0.3, color=COLORS['primary'])

    set_plot_style(ax1, 'Evolução do Número de Evacuados',
                   'Hora', 'Número Total de Evacuados')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)

    # Gráfico de taxa de evacuação
    ax2.bar(times, rate, color=COLORS['primary'], alpha=0.7)
    set_plot_style(ax2, 'Taxa de Evacuação por Intervalo',
                   'Hora', 'Pessoas Evacuadas no Intervalo')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)

    # Adicionar valores nas barras
    for i, v in enumerate(rate):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)


def _plot_survival_factors(data):
    """Apresenta análise detalhada dos factores de sobrevivência"""
    fig, ax = plt.subplots(figsize=(12, 6))

    survival_by_class = pd.crosstab(data['Pclass'], data['Survived'], normalize='index') * 100
    x = np.arange(len(['1.ª Classe', '2.ª Classe', '3.ª Classe']))
    width = 0.35

    bars1 = ax.bar(x - width / 2, survival_by_class[0],
                   width, label='Não Sobreviveu', color=COLORS['negative'])
    bars2 = ax.bar(x + width / 2, survival_by_class[1],
                   width, label='Sobreviveu', color=COLORS['primary'])

    set_plot_style(
        ax,
        'Taxa de Sobrevivência por Classe',
        'Classe',
        'Percentagem (%)'
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


def _plot_age_fare_distribution(data):
    """Apresenta distribuição de idade e tarifa por sobrevivência"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Distribuição de idade
    sns.kdeplot(data=data[data['Survived'] == 0], x='Age',
                label='Não Sobreviventes', color=COLORS['negative'], ax=ax1)
    sns.kdeplot(data=data[data['Survived'] == 1], x='Age',
                label='Sobreviventes', color=COLORS['primary'], ax=ax1)

    ax1.set_title('Distribuição de Idade por Sobrevivência',
                  pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Idade (anos)')
    ax1.set_ylabel('Densidade')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Distribuição de tarifa
    sns.kdeplot(data=data[data['Survived'] == 0], x='Fare',
                label='Não Sobreviventes', color=COLORS['negative'], ax=ax2)
    sns.kdeplot(data=data[data['Survived'] == 1], x='Fare',
                label='Sobreviventes', color=COLORS['primary'], ax=ax2)

    ax2.set_title('Distribuição de Tarifa por Sobrevivência',
                  pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tarifa (£)')
    ax2.set_ylabel('Densidade')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)


def _show_insights(data):
    """Apresenta informações detalhadas da análise de sobrevivência"""
    st.divider()
    st.markdown("### Principais Conclusões sobre Sobrevivência")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Estatísticas Principais**")
        survivors = len(data[data['Survived'] == 1])
        total = len(data)
        avg_age_survivors = data[data['Survived'] == 1]['Age'].mean()
        avg_fare_survivors = data[data['Survived'] == 1]['Fare'].mean()

        st.markdown(f"""
        - **Sobrevivência Geral**:
            - Total: {total} passageiros
            - Sobreviventes: {survivors} ({format_percentage(survivors / total * 100)})
            - Não sobreviventes: {total - survivors} ({format_percentage((total - survivors) / total * 100)})
        - **Perfil dos Sobreviventes**:
            - Idade média: {avg_age_survivors:.1f} anos
            - Tarifa média: £{avg_fare_survivors:.2f}
            - Rácio de mortalidade: {((total - survivors) / survivors):.2f}
        """)

    with col2:
        st.markdown("#### **Factores Críticos**")
        st.markdown("""
        - **Capacidade de Salvamento**:
            - 20 botes disponíveis
            - 1.178 lugares totais
            - 28,9% de utilização
        - **Condições do Desastre**:
            - Tempo total: 2h 40min
            - Temperatura: -2°C
            - Sobrevivência na água: 15-30 min
        """)

    st.divider()

    st.markdown("### ⏰ Fases do Desastre")
    st.markdown("""
    **Cronologia Crítica:**
    - **23:40-00:15**: Fase inicial de avaliação
    - **00:15-01:15**: Evacuação principal
    - **01:15-02:20**: Fase final e naufrágio
    - **Tempo médio de evacuação**: ~8 min/bote
    """)

    st.divider()

    st.markdown("### 🔍 Factores Determinantes")
    st.markdown("""
    **Elementos Críticos:**
    - Capacidade limitada dos botes
    - Tempo limitado de evacuação
    - Condições ambientais severas
    - Organização da evacuação
    """)

    st.divider()

    st.markdown("### 📊 Padrões Identificados")
    st.markdown("""
    **Observações Principais:**
    - Maior sobrevivência nas classes superiores
    - Impacto significativo da tarifa paga
    - Relação idade-sobrevivência
    - Eficiência da evacuação por fase
    """)

    st.divider()

    # Legado Histórico
    st.markdown("### 📚 Legado e Impacto Histórico")
    st.markdown("""
    **Mudanças e Consequências:**
    - Revisão completa das regulamentações marítimas
    - Novos protocolos de segurança implementados
    - Maior conscientização sobre segurança marítima
    - Influência na cultura e história moderna
    """)

    st.divider()