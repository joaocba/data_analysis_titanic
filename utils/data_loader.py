# utils/data_loader.py
import pandas as pd
from config import DATASET_URL
import streamlit as st


def load_data():
    """Carrega e faz cache dos dados do Titanic"""

    @st.cache_data
    def _load_data():
        return pd.read_csv(DATASET_URL)

    return _load_data()
