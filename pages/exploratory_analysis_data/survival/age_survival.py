import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import get_color_palette, COLORS, set_plot_style, format_percentage

def show(data):
    st.markdown("### Análise de Sobrevivência por Faixa Etária")

    st.markdown("""
    Nesta secção, vamos explorar como a idade influenciou as possibilidades de sobrevivência dos passageiros do Titanic. 
    Poderá visualizar as taxas de sobrevivência por faixa etária, compreender os padrões de sobrevivência 
    entre diferentes grupos etários e identificar fatores que podem ter influenciado estes resultados.
    """)

    data['FaixaEtaria'] = data['Age'].apply(categorize_age)

    survival_by_age = pd.crosstab(data['FaixaEtaria'], data['Survived'])
    survival_rate_age = pd.crosstab(data['FaixaEtaria'], data['Survived'], normalize='index') * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Estatísticas por Faixa Etária")
        stats_df = _create_age_stats_df(data)
        st.table(stats_df)

    with col2:
        st.markdown("#### Taxa de Sobrevivência")
        survival_stats_df = _create_survival_stats_df(survival_by_age, survival_rate_age)
        st.table(survival_stats_df)

    st.markdown("### Visualizações e Perspetivas")
    _show_visualizations(data, survival_rate_age)
    _show_insights(data)

def _create_age_stats_df(data):
    age_groups = ['Criança (0-17)', 'Adulto (18-64)', 'Idoso (65+)']
    stats = []

    for group in age_groups:
        group_data = data[data['FaixaEtaria'] == group]
        stats.append({
            'Faixa Etária': group,
            'Total': len(group_data),
            'Idade Média': f"{group_data['Age'].mean():.1f} anos",
            'Idade Mediana': f"{group_data['Age'].median():.1f} anos",
            'Desvio Padrão': f"{group_data['Age'].std():.1f} anos"
        })

    return pd.DataFrame(stats)

def _create_survival_stats_df(survival_by_age, survival_rate_age):
    return pd.DataFrame({
        'Faixa Etária': survival_by_age.index,
        'Total': survival_by_age.sum(axis=1),
        'Sobreviventes': survival_by_age[1],
        'Não Sobreviventes': survival_by_age[0],
        'Taxa Sobrevivência': survival_rate_age[1].apply(format_percentage)
    }).sort_values(by='Taxa Sobrevivência', ascending=False)

def _show_visualizations(data, survival_rate_age):
    st.markdown("#### Visualizações da Sobrevivência por Idade")
    tab1, tab2, tab3 = st.tabs(["Taxa de Sobrevivência", "Distribuição por Idade", "Análise Detalhada"])

    with tab1:
        _plot_survival_rate(survival_rate_age)
    with tab2:
        _plot_age_distribution(data)
    with tab3:
        _plot_detailed_analysis(data)

def _plot_survival_rate(survival_rate_age):
    fig, ax = plt.subplots(figsize=(12, 6))
    survival_rate_sorted = survival_rate_age.sort_values(by=1, ascending=False)

    colors = get_color_palette(20)
    bars = survival_rate_sorted.plot(
        kind='bar',
        color=[colors[3], colors[0]],
        edgecolor='white',
        linewidth=2,
        width=0.8,
        ax=ax
    )

    set_plot_style(
        ax,
        'Taxa de Sobrevivência por Faixa Etária',
        'Faixa Etária',
        'Percentagem (%)'
    )

    plt.legend(['Não Sobreviveu', 'Sobreviveu'],
               frameon=True,
               facecolor='white',
               edgecolor='none')

    for container in bars.containers:
        bars.bar_label(container,
                       fmt='%.1f%%',
                       label_type='center',
                       color='black',
                       fontweight='bold')

    ax.set_xticklabels(survival_rate_sorted.index, rotation=0)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def _plot_age_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = get_color_palette(20)
    sns.kdeplot(data=data[data['Survived'] == 0], x='Age',
                label='Não Sobreviventes', color=colors[3], ax=ax)
    sns.kdeplot(data=data[data['Survived'] == 1], x='Age',
                label='Sobreviventes', color=colors[0], ax=ax)

    set_plot_style(
        ax,
        'Distribuição de Idade: Sobreviventes vs. Não Sobreviventes',
        'Idade (anos)',
        'Densidade'
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def _plot_detailed_analysis(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    survival_by_group = data.groupby('FaixaEtaria')['Survived'].agg(['count', 'mean'])
    survival_by_group['mean'] *= 100

    colors = get_color_palette(20)
    bars = plt.bar(survival_by_group.index,
                   survival_by_group['mean'],
                   color=colors[0])

    set_plot_style(
        ax,
        'Probabilidade de Sobrevivência por Faixa Etária',
        'Faixa Etária',
        'Probabilidade de Sobrevivência (%)'
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

def _show_insights(data):
    st.divider()
    st.markdown("### Principais Perspetivas sobre Idade e Sobrevivência")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Análise por Grupos**")
        children = data[data['Age'] <= 17]
        adults = data[(data['Age'] > 17) & (data['Age'] <= 64)]
        elderly = data[data['Age'] > 64]

        st.markdown(f"""
        - **Crianças (0-17)**:
            - Total: {len(children)}
            - Sobreviventes: {len(children[children['Survived'] == 1])}
            - Taxa: {format_percentage(children['Survived'].mean() * 100)}
        - **Adultos (18-64)**:
            - Total: {len(adults)}
            - Sobreviventes: {len(adults[adults['Survived'] == 1])}
            - Taxa: {format_percentage(adults['Survived'].mean() * 100)}
        - **Idosos (65+)**:
            - Total: {len(elderly)}
            - Sobreviventes: {len(elderly[elderly['Survived'] == 1])}
            - Taxa: {format_percentage(elderly['Survived'].mean() * 100)}
        """)

    with col2:
        st.markdown("#### **Padrões Observados**")
        st.markdown("""
        - **Priorização**:
            - Crianças tiveram prioridade
            - Política "mulheres e crianças primeiro"
        - **Fatores Físicos**:
            - Mobilidade diferenciada por idade
            - Resistência às condições adversas
        - **Localização no Navio**:
            - Diferentes classes por faixa etária
            - Acesso aos botes salva-vidas
        """)

    st.divider()

    st.markdown("### 📊 Conclusões Principais")
    st.markdown("""
    **Principais Observações:**
    - A idade foi um fator significativo na sobrevivência
    - Crianças tiveram as maiores possibilidades de sobrevivência
    - Existe uma correlação negativa entre idade e sobrevivência
    - Fatores sociais e físicos influenciaram os resultados
    """)

    st.divider()

    st.markdown("### 🔍 Implicações Históricas")
    st.markdown("""
    **Relevância Histórica:**
    - Os dados refletem os valores sociais da época
    - A priorização de grupos vulneráveis era uma prática estabelecida
    - O desastre influenciou futuras políticas de segurança marítima
    - O padrão de sobrevivência por idade gerou lições importantes
    """)

    st.divider()

def categorize_age(age):
    if pd.isna(age):
        return "Não Informado"
    elif age <= 17:
        return "Criança (0-17)"
    elif age <= 64:
        return "Adulto (18-64)"
    else:
        return "Idoso (65+)"