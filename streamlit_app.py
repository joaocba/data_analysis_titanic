import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Análise do Titanic",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

from config import APP_CONFIG
from utils.data_loader import load_data
from pages import intro, initial_analysis, data_cleaning, exploratory_analysis, modeling, conclusions

def main():
    pages = {
        "1. Introdução": intro.show,
        "2. Integração e Análise Inicial": initial_analysis.show,
        "3. Limpeza e Transformação": data_cleaning.show,
        "4. Análise Exploratória": exploratory_analysis.show,
        "5. Modelação": modeling.show,
    }

    with st.sidebar:
        st.title("Navegação")
        st.divider()
        selection = st.radio("", list(pages.keys()), label_visibility="collapsed")

    with st.container():
        if selection == "4. Análise Exploratória":
            data = pages[selection]()
            if data is not None:
                st.session_state['processed_data'] = data
        elif selection == "5. Modelação":
            if 'processed_data' in st.session_state:
                pages[selection](st.session_state['processed_data'])
            else:
                st.error("É necessário executar primeiro a Análise Exploratória para processar os dados.")
        else:
            pages[selection]()

if __name__ == "__main__":
    main()