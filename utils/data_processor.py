# utils/data_processor.py
import pandas as pd
import numpy as np

def clean_data(df):
    """Limpa e processa os dados do Titanic"""
    df_clean = df.copy()

    # Tratamento de valores ausentes
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Cabin'] = df_clean['Cabin'].fillna('Unknown')
    df_clean = df_clean.dropna(subset=['Embarked'])

    # Codificação de variáveis
    df_clean['Sex'] = (df_clean['Sex'] == 'female').astype(int)

    # Criação de novas features
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch']
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 0).astype(int)

    return df_clean


def prepare_features(df):
    """Prepara features para modelagem"""
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    X = df[features].copy()
    y = df['Survived']
    return X, y