import streamlit as st
from utils.data_loader import load_data
from utils.data_processor import clean_data
from pages.exploratory_analysis_data.distributions import (
    age_analysis,
    fare_analysis,
    family_analysis,
    gender_analysis
)
from pages.exploratory_analysis_data.survival import (
    general_survival,
    age_survival,
    gender_survival,
    class_survival,
    family_survival,
    port_survival,
    combined_survival
)
from pages.exploratory_analysis_data import correlation_analysis

def show():
    st.title("📊 Análise Exploratória")

    st.markdown("""
    Esta página oferece diversas análises exploratórias sobre os dados do Titanic. 
    Pode explorar:
    - 📈 Distribuições de variáveis
    - ⚖️ Relações de sobrevivência
    - 🔍 Correlações entre variáveis
    """)

    data = load_data()
    data = clean_data(data)

    tab1, tab2, tab3 = st.tabs([
        "Distribuições",
        "Relações de Sobrevivência",
        "Correlações"
    ])

    with tab1:
        _show_distributions_analysis(data)

    with tab2:
        _show_survival_analysis(data)

    with tab3:
        correlation_analysis.show(data)

    return data

def _show_distributions_analysis(data):
    st.subheader("Distribuições de Variáveis")
    st.markdown("""
    Escolha uma das distribuições abaixo para visualizar a análise detalhada.
    """)

    distribution_type = st.selectbox(
        "Escolha a distribuição para analisar:",
        ["Idade", "Tarifas", "Dimensão das Famílias", "Sexo"]
    )

    if distribution_type == "Idade":
        age_analysis.show(data)
    elif distribution_type == "Tarifas":
        fare_analysis.show(data)
    elif distribution_type == "Dimensão das Famílias":
        family_analysis.show(data)
    elif distribution_type == "Sexo":
        gender_analysis.show(data)

def _show_survival_analysis(data):
    st.subheader("Análise de Sobrevivência")
    st.markdown("""
    Escolha o tipo de análise de sobrevivência para compreender como diferentes características influenciam a taxa de sobrevivência.
    """)

    survival_type = st.selectbox(
        "Escolha o tipo de análise de sobrevivência:",
        [
            "Geral",
            "Por Faixa Etária",
            "Por Sexo",
            "Por Classe",
            "Por Dimensão de Família",
            "Por Porto de Embarque",
            "Análise Combinada"
        ]
    )

    if survival_type == "Geral":
        general_survival.show(data)
    elif survival_type == "Por Faixa Etária":
        age_survival.show(data)
    elif survival_type == "Por Sexo":
        gender_survival.show(data)
    elif survival_type == "Por Classe":
        class_survival.show(data)
    elif survival_type == "Por Dimensão de Família":
        family_survival.show(data)
    elif survival_type == "Por Porto de Embarque":
        port_survival.show(data)
    else:
        combined_survival.show(data)

if __name__ == "__main__":
    show()