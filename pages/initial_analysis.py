import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data
from config import PLOT_CONFIG

def show_data_overview():
    data = load_data()
    st.markdown("### 📊 Visão Geral dos Dados")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Registos", data.shape[0])
    with col2:
        st.metric("🛠️ Variáveis", data.shape[1])
    with col3:
        st.metric("💡 Taxa de Sobrevivência", f"{(data['Survived'].mean() * 100):.1f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Passageiros com Família",
                  f"{(((data['SibSp'] + data['Parch']) > 0).mean() * 100):.1f}%")
    with col2:
        st.metric("💰 Tarifa Média",
                  f"£{data['Fare'].mean():.2f}")
    with col3:
        st.metric("👶 Idade Média",
                  f"{data['Age'].mean():.1f} anos")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👨 Homens",
                  f"{(data['Sex'] == 'male').mean() * 100:.1f}%",
                  f"Total: {(data['Sex'] == 'male').sum()}")
    with col2:
        st.metric("👩 Mulheres",
                  f"{(data['Sex'] == 'female').mean() * 100:.1f}%",
                  f"Total: {(data['Sex'] == 'female').sum()}")
    with col3:
        st.metric("🎫 Classe Mais Comum",
                  f"{data['Pclass'].mode().iloc[0]}ª Classe",
                  f"({(data['Pclass'] == data['Pclass'].mode().iloc[0]).mean() * 100:.1f}% dos passageiros)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚢 Portos de Embarque")
        port_counts = data['Embarked'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=port_counts.index, y=port_counts.values)
        plt.title('Distribuição dos Portos de Embarque')
        st.pyplot(fig)

    with col2:
        st.markdown("#### 👨‍👩‍👧‍👦 Distribuição de Família")
        family_size = data['SibSp'] + data['Parch']
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(family_size, bins=range(max(family_size) + 2), discrete=True)
        plt.title('Tamanho das Famílias')
        st.pyplot(fig)

    st.markdown("""
    #### 🔍 Resumo Rápido:
    - A maioria dos passageiros era do sexo masculino
    - A idade média dos passageiros era relativamente jovem
    - Aproximadamente um terço dos passageiros sobreviveu
    - A maioria dos passageiros viajava sozinha ou com família pequena
    - Os preços das passagens variavam significativamente
    """)

def show_data_sample():
    data = load_data()
    st.markdown("### 🔍 Amostra dos Dados")

    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.slider("Número de linhas:", 5, 50, 10)
    with col2:
        sort_by = st.selectbox("Ordenar por:", ['Índice'] + list(data.columns))

    if sort_by != 'Índice':
        data_display = data.sort_values(by=sort_by).head(n_rows)
    else:
        data_display = data.head(n_rows)

    st.dataframe(data_display)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Transferir Conjunto de Dados Completo (CSV)",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='titanic_data.csv',
            mime='text/csv',
        )
    with col2:
        st.download_button(
            label="⬇️ Transferir Amostra (CSV)",
            data=data_display.to_csv(index=False).encode('utf-8'),
            file_name='titanic_sample.csv',
            mime='text/csv',
        )

def show_data_types():
    data = load_data()
    st.markdown("### 🧩 Tipos de Dados")

    var_info = pd.DataFrame({
        'Tipo': data.dtypes.astype(str),
        'Não Nulos': data.count(),
        'Nulos': data.isnull().sum(),
        '% Nulos': (data.isnull().sum() / len(data) * 100).round(2),
        'Valores Únicos': data.nunique(),
        'Primeiro Valor': data.iloc[0],
        'Último Valor': data.iloc[-1],
        'Memória (KB)': data.memory_usage(deep=True) / 1024
    })

    styled_var_info = var_info.style.background_gradient(subset=['% Nulos'], cmap='RdYlGn_r') \
        .background_gradient(subset=['Valores Únicos'], cmap='YlOrRd')

    st.dataframe(styled_var_info, use_container_width=True)

    st.markdown("#### 📊 Distribuição dos Tipos de Dados")
    type_counts = data.dtypes.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=type_counts.index.astype(str), y=type_counts.values)
    plt.title('Distribuição dos Tipos de Dados')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def show_missing_values():
    data = load_data()
    st.markdown("### ⚠️ Análise de Valores em Falta")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Mapa de Valores em Falta')
        st.pyplot(fig)

    with col2:
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
        sns.barplot(x=missing.values, y=missing.index)
        plt.title('Quantidade de Valores em Falta por Variável')
        st.pyplot(fig)

    missing_stats = pd.DataFrame({
        'Valores em Falta': data.isnull().sum(),
        'Percentagem (%)': (data.isnull().sum() / len(data) * 100).round(2),
        'Tipo de Dado': data.dtypes.astype(str)
    }).sort_values('Valores em Falta', ascending=False)

    st.dataframe(missing_stats.style.background_gradient(subset=['Percentagem (%)'], cmap='RdYlGn_r'),
                 use_container_width=True)

def show_basic_stats():
    data = load_data()
    st.markdown("### 📈 Estatísticas Básicas")

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_col = st.selectbox("Selecione uma variável numérica:", numeric_cols)

    with col2:
        plot_type = st.selectbox("Tipo de visualização:",
                                 ["Histograma", "Diagrama de Caixa", "Diagrama de Violino"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📊 Estatísticas Descritivas")
        stats_df = pd.DataFrame({
            'Estatística': [
                'Contagem', 'Média', 'Desvio Padrão', 'Mínimo', 'Q1 (25%)',
                'Mediana', 'Q3 (75%)', 'Máximo', 'Assimetria', 'Curtose'
            ],
            'Valor': [
                data[selected_col].count(),
                data[selected_col].mean(),
                data[selected_col].std(),
                data[selected_col].min(),
                data[selected_col].quantile(0.25),
                data[selected_col].median(),
                data[selected_col].quantile(0.75),
                data[selected_col].max(),
                data[selected_col].skew(),
                data[selected_col].kurtosis()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.markdown("#### 📉 Distribuição")
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

        if plot_type == "Histograma":
            sns.histplot(data=data, x=selected_col, kde=True)
        elif plot_type == "Diagrama de Caixa":
            sns.boxplot(data=data, y=selected_col)
        else:
            sns.violinplot(data=data, y=selected_col)

        plt.title(f'{plot_type} de {selected_col}')
        st.pyplot(fig)

        Q1 = data[selected_col].quantile(0.25)
        Q3 = data[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[selected_col] < (Q1 - 1.5 * IQR)) |
                        (data[selected_col] > (Q3 + 1.5 * IQR))][selected_col]

        if not outliers.empty:
            st.markdown(f"**Valores Atípicos Detetados:** {len(outliers)} valores")

def show():
    st.title("📝 Integração e Análise Inicial")

    st.markdown("""
    Esta secção apresenta uma análise inicial e completa dos dados do Titanic, incluindo:
    - 📊 Visão geral detalhada do conjunto de dados
    - 🔍 Amostra dos dados com opções de ordenação
    - 🧩 Análise aprofundada dos tipos de dados
    - ⚠️ Análise detalhada de valores em falta
    - 📈 Estatísticas avançadas e visualizações
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Visão Geral",
        "Amostra",
        "Tipos de Dados",
        "Valores em Falta",
        "Estatísticas"
    ])

    with tab1:
        show_data_overview()
    with tab2:
        show_data_sample()
    with tab3:
        show_data_types()
    with tab4:
        show_missing_values()
    with tab5:
        show_basic_stats()

if __name__ == "__main__":
    show()