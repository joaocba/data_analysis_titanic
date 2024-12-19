import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data

def show_initial_state(data):
    st.subheader("Estado Inicial dos Dados")
    st.metric("Número de Registos Inicial", len(data))

    st.subheader("Valores em Falta (Antes do Tratamento)")

    missing_values = pd.DataFrame({
        'Coluna': data.columns,
        'Valores em Falta': data.isnull().sum(),
        'Percentagem (%)': (data.isnull().sum() / len(data) * 100).round(2)
    }).reset_index(drop=True)

    styled_missing_values = (
        missing_values.style
        .background_gradient(subset=['Valores em Falta'], cmap='Reds')
        .format({'Percentagem (%)': "{:.2f}%"})
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#f7f7f7'), ('font-weight', 'bold')]},
            {'selector': 'tbody td', 'props': [('text-align', 'center')]},
        ])
    )

    st.dataframe(styled_missing_values, use_container_width=True)

def process_missing_values(data):
    st.subheader("Tratamento de Valores em Falta")

    st.subheader("Variável 'Age'")
    age_median = data['Age'].median()
    data['Age'] = data['Age'].fillna(age_median)
    st.metric("Idade Mediana Utilizada", f"{age_median:.2f}")

    st.subheader("Variável 'Embarked'")
    n_missing_embarked = data['Embarked'].isnull().sum()
    data = data.dropna(subset=['Embarked'])
    st.metric("Registos Removidos (Embarked)", n_missing_embarked)

    st.subheader("Variável 'Cabin'")
    n_missing_cabin = data['Cabin'].isnull().sum()
    data.loc[:, 'Cabin'] = data['Cabin'].fillna('Desconhecido')
    st.metric("Valores Substituídos por 'Desconhecido'", n_missing_cabin)

    st.metric("Número de Registos Após Tratamento", len(data))
    return data

def encode_categorical_variables(data):
    st.subheader("Codificação de Variáveis Categóricas")

    st.subheader("Variável 'Sex'")
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    st.metric("Masculino (0)", data['Sex'].value_counts()[0])
    st.metric("Feminino (1)", data['Sex'].value_counts()[1])

    st.subheader("Variável 'Embarked'")
    data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked', drop_first=False)
    dummy_columns = [col for col in data.columns if 'Embarked_' in col]
    st.write("**Novas Colunas Criadas:**", ', '.join(dummy_columns))
    st.dataframe(data.head())

    return data

import matplotlib.pyplot as plt

def create_derived_variables(data):
    st.subheader("Criação de Variáveis Derivadas")

    st.subheader("Variável 'FamilySize'")
    data['FamilySize'] = data['SibSp'] + data['Parch']
    st.write("**Exemplo de Registos com a Nova Variável 'FamilySize':**")
    st.dataframe(data[['SibSp', 'Parch', 'FamilySize']].head())

    st.subheader("Variável 'IsAlone'")
    data['IsAlone'] = (data['FamilySize'] == 0).astype(int)
    st.write("**Exemplo de Registos com a Nova Variável 'IsAlone':**")
    st.dataframe(data[['FamilySize', 'IsAlone']].head())

    st.metric("Taxa de Passageiros Sozinhos", f"{(data['IsAlone'].mean() * 100):.2f}%")

    return data

def create_age_category(data):
    st.subheader("Criação de Variável 'Faixa Etária'")

    def categorize_age(age):
        if age <= 17:
            return 'Criança (0-17)'
        elif age <= 64:
            return 'Adulto (18-64)'
        else:
            return 'Idoso (65+)'

    data['FaixaEtaria'] = data['Age'].apply(categorize_age)

    st.write("**Exemplo de Registos com a Nova Variável 'Faixa Etária':**")
    st.dataframe(data[['Age', 'FaixaEtaria']].head())

    return data

def show_final_summary(data):
    st.subheader("Resumo das Transformações")
    st.markdown("""
    #### Destaques:
    - Valores em falta tratados com mediana ou substituição
    - Registos inválidos removidos
    - Variáveis categóricas codificadas
    - Variáveis derivadas criadas (FamilySize, IsAlone, FaixaEtaria)
    """)

    st.download_button(
        label="📥 Transferir Dados Processados (CSV)",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name='titanic_processed.csv',
        mime='text/csv',
    )

def show():
    st.title("🧹 Limpeza e Transformação de Dados")

    st.markdown("""
    Esta secção é dedicada à limpeza e transformação dos dados do Titanic, abordando as seguintes etapas:
    - 🧹 Tratamento de valores em falta
    - 🔄 Codificação de variáveis categóricas
    - 📊 Criação de variáveis derivadas
    - ⚕️ Criação de variável de faixa etária
    """)

    data = load_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Valores em Falta",
        "Codificação de Variáveis",
        "Variáveis Derivadas",
        "Faixa Etária",
        "Resumo Final"
    ])

    with tab1:
        show_initial_state(data)
    with tab2:
        data = process_missing_values(data)
    with tab3:
        data = encode_categorical_variables(data)
    with tab4:
        data = create_derived_variables(data)
        data = create_age_category(data)
    with tab5:
        show_final_summary(data)

if __name__ == "__main__":
    show()