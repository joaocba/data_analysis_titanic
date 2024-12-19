import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder


def show(data):
    def ensure_dummies(data):
        if 'Embarked_C' not in data.columns or 'Embarked_Q' not in data.columns or 'Embarked_S' not in data.columns:
            data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked')
        return data

    # Criar cópia dos dados para não modificar o original
    data = data.copy()

    # Converter 'Survived' para valores numéricos
    le = LabelEncoder()
    if 'Survived' in data.columns:
        if data['Survived'].dtype == 'object':
            data['Survived'] = le.fit_transform(data['Survived'])

    # Garantir que as variáveis dummy estão presentes
    data = ensure_dummies(data)

    # Converter 'Sex' para numérico se necessário
    if 'Sex' in data.columns and data['Sex'].dtype == 'object':
        data['Sex'] = le.fit_transform(data['Sex'])

    # Definir as características para o modelo
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

    # Verificar características
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        st.error(f"Colunas em falta no conjunto de dados: {', '.join(missing_features)}")
        return

    # Preparar dados
    X = data[features]
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cabeçalho com título e descrição
    st.title("🤖 Modelação Preditiva")

    st.markdown("""
    Esta análise utiliza dois modelos de aprendizagem automática diferentes para prever a sobrevivência 
    dos passageiros do Titanic. Compare os resultados e explore as conclusões geradas por cada modelo.

    #### 🎯 Objectivos:
    - Avaliar o desempenho de diferentes modelos
    - Identificar os factores mais importantes para sobrevivência
    - Comparar métricas e resultados
    """)

    # Criar separadores principais com ícones e descrições
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral dos Dados",
        "🌳 Árvore de Decisão",
        "🎯 KNN",
        "📈 Comparação e Conclusões"
    ])

    # Separador 1: Visão Geral dos Dados
    with tab1:
        st.header("📊 Visão Geral dos Dados de Treino e Teste")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="📝 Total de Amostras",
                value=len(X),
                delta=f"{len(X_train)} treino / {len(X_test)} teste"
            )

        with col2:
            survival_rate = (y == 1).mean() * 100
            st.metric(
                label="💫 Taxa de Sobrevivência",
                value=f"{survival_rate:.1f}%",
                delta=f"{len(features)} características utilizadas"
            )

        with col3:
            class_balance = f"{(y_train == 1).mean() * 100:.1f}% / {(y_test == 1).mean() * 100:.1f}%"
            st.metric(
                label="⚖️ Equilíbrio das Classes",
                value=class_balance
            )

        st.subheader("🔍 Características Utilizadas no Modelo")

        feature_df = pd.DataFrame({
            'Característica': features,
            'Tipo': [X[f].dtype for f in features],
            'Valores Únicos': [X[f].nunique() for f in features],
            'Valores em Falta': [X[f].isnull().sum() for f in features]
        })

        st.dataframe(
            feature_df.style.background_gradient(subset=['Valores Únicos'], cmap='YlOrRd')
            .background_gradient(subset=['Valores em Falta'], cmap='RdYlGn_r'),
            hide_index=True
        )

# Separador 2: Árvore de Decisão
    with tab2:
        st.header("🌳 Modelo: Árvore de Decisão")

        # Parâmetros ajustáveis na mesma página
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.slider("Profundidade Máxima", 1, 20, 5)
        with col2:
            min_samples_split = st.slider("Mínimo para Divisão", 2, 20, 2)

        # Treino com parâmetros ajustáveis
        dt_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)

        # Métricas
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Métricas de Desempenho")
            dt_accuracy = accuracy_score(y_test, dt_pred)
            dt_f1 = f1_score(y_test, dt_pred)

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("🎯 Precisão", f"{dt_accuracy:.3f}")
            with metrics_col2:
                st.metric("📈 Pontuação F1", f"{dt_f1:.3f}")

        with col2:
            st.subheader("🔄 Validação Cruzada")
            cv_scores = cross_val_score(dt_model, X, y, cv=5)
            st.metric("🔄 Média VC", f"{cv_scores.mean():.3f}")
            st.metric("📊 Desvio Padrão VC", f"{cv_scores.std():.3f}")

        # Matriz de Confusão
        st.subheader("📊 Matriz de Confusão")
        dt_conf_matrix = confusion_matrix(y_test, dt_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                    xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                    yticklabels=['Não Sobreviveu', 'Sobreviveu'])
        plt.title('Matriz de Confusão - Árvore de Decisão', pad=20)
        st.pyplot(fig)

        # Importância das Características
        st.subheader("🔍 Importância das Características")
        feature_importance = pd.DataFrame({
            'Característica': features,
            'Importância': dt_model.feature_importances_
        }).sort_values('Importância', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(feature_importance['Característica'], feature_importance['Importância'])
        ax.set_title('Importância das Características na Árvore de Decisão')

        for i, v in enumerate(feature_importance['Importância']):
            ax.text(v, i, f'{v:.3f}', va='center')

        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(0, max(feature_importance['Importância'])))
        for bar, importance in zip(bars, feature_importance['Importância']):
            bar.set_color(sm.to_rgba(importance))

        plt.tight_layout()
        st.pyplot(fig)

    # Separador 3: KNN
    with tab3:
        st.header("🎯 Modelo: K-Vizinhos Mais Próximos (KNN)")

        # Parâmetros ajustáveis na mesma página
        col1, col2 = st.columns(2)
        with col1:
            n_neighbors = st.slider("Número de Vizinhos (K)", 1, 20, 5)
        with col2:
            weights = st.selectbox("Peso dos Vizinhos", ['uniforme', 'distância'])
            # Mapear opções em português para valores em inglês
            weights_map = {'uniforme': 'uniform', 'distância': 'distance'}
            weights = weights_map[weights]

        # Treino com parâmetros ajustáveis
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)

        # Métricas
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Métricas de Desempenho")
            knn_accuracy = accuracy_score(y_test, knn_pred)
            knn_f1 = f1_score(y_test, knn_pred)

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("🎯 Precisão", f"{knn_accuracy:.3f}")
            with metrics_col2:
                st.metric("📈 Pontuação F1", f"{knn_f1:.3f}")

        with col2:
            st.subheader("🔄 Validação Cruzada")
            cv_scores = cross_val_score(knn_model, X, y, cv=5)
            st.metric("🔄 Média VC", f"{cv_scores.mean():.3f}")
            st.metric("📊 Desvio Padrão VC", f"{cv_scores.std():.3f}")

        # Matriz de Confusão
        st.subheader("📊 Matriz de Confusão")
        knn_conf_matrix = confusion_matrix(y_test, knn_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                    xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                    yticklabels=['Não Sobreviveu', 'Sobreviveu'])
        plt.title('Matriz de Confusão - KNN', pad=20)
        st.pyplot(fig)

    # Separador 4: Comparação e Conclusões
    with tab4:
        st.header("📈 Comparação dos Modelos e Conclusões")

        # Comparação visual das métricas
        comparison_df = pd.DataFrame({
            'Modelo': ['Árvore de Decisão', 'KNN'],
            'Precisão': [dt_accuracy, knn_accuracy],
            'Pontuação F1': [dt_f1, knn_f1]
        })

        # Gráfico de comparação
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison_df['Modelo']))
        width = 0.35

        bars1 = ax.bar(x - width / 2, comparison_df['Precisão'], width, label='Precisão',
                       color='skyblue')
        bars2 = ax.bar(x + width / 2, comparison_df['Pontuação F1'], width, label='Pontuação F1',
                       color='lightcoral')

        ax.set_ylabel('Pontuação')
        ax.set_title('Comparação de Métricas entre Modelos')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Modelo'])
        ax.legend()

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(bars1)
        autolabel(bars2)

        plt.tight_layout()
        st.pyplot(fig)

        # Conclusões em markdown
        st.markdown("""
        #### 🎯 Desempenho Global
        - A Árvore de Decisão alcançou uma precisão de {:.1f}% vs {:.1f}% do KNN
        - A Pontuação F1 mostra uma diferença de {:.1f} pontos entre os modelos
        - A estabilidade da Árvore de Decisão é {}
        """.format(
            dt_accuracy * 100,
            knn_accuracy * 100,
            abs(dt_f1 - knn_f1) * 100,
            "superior" if dt_accuracy > knn_accuracy else "inferior"
        ))

        st.markdown("""
        #### 🔍 Análise de Características
        - As 3 características mais importantes são: {}
        - Características com menor impacto: {}
        - Recomendação: considerar remover características com importância < 5%
        """.format(
            ", ".join(feature_importance['Característica'].tail(3).tolist()),
            ", ".join(feature_importance['Característica'].head(2).tolist())
        ))

        st.divider()

        st.subheader("📥 Download dos Resultados")

        results_dict = {
            'Métrica': ['Precisão', 'Pontuação F1', 'Média da Validação Cruzada'],
            'Árvore de Decisão': [dt_accuracy, dt_f1, cv_scores.mean()],
            'KNN': [knn_accuracy, knn_f1, cv_scores.mean()]
        }
        results_df = pd.DataFrame(results_dict)

        st.download_button(
            "Download Resultados",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='resultados.csv',
            mime='text/csv'
        )

        st.divider()