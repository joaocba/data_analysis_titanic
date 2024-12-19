import streamlit as st

def show():
    st.title("🚢 Projeto Titanic - Análise de Sobrevivência")

    with st.container():
        st.markdown("### 🌟 Objetivo do Projeto")
        st.info(
            "Este projeto visa analisar os dados dos passageiros do Titanic para "
            "identificar padrões de sobrevivência e desenvolver modelos preditivos. "
            "O principal objetivo é compreender quais características dos passageiros "
            "influenciaram as suas possibilidades de sobrevivência durante o desastre."
        )

    st.markdown("### 🎯 Objetivos Específicos")
    st.write(
        """
        - Identificar padrões demográficos entre os sobreviventes
        - Analisar a influência de fatores socioeconómicos na sobrevivência
        - Desenvolver modelos preditivos para estimar probabilidades de sobrevivência
        - Determinar as variáveis mais relevantes para a sobrevivência
        - Avaliar a precisão dos modelos desenvolvidos
        """
    )

    st.divider()

    st.markdown("### 🛠️ Metodologia")
    st.expander("1️⃣ Preparação dos Dados").write(
        """
        - Tratamento de valores em falta
        - Codificação de variáveis categóricas
        - Criação de novas características
        - Limpeza e validação dos dados
        - Normalização de variáveis numéricas
        """
    )
    st.expander("2️⃣ Análise Exploratória").write(
        """
        - Estudo das distribuições das variáveis
        - Identificação de padrões e correlações
        - Análise de sobrevivência por grupos
        - Visualização de relações importantes
        - Testes estatísticos relevantes
        """
    )
    st.expander("3️⃣ Modelação").write(
        """
        - Implementação de diferentes algoritmos
        - Avaliação e comparação de desempenho
        - Seleção do melhor modelo
        - Otimização de hiperparâmetros
        - Validação cruzada dos resultados
        """
    )

    st.markdown("### 📊 Sobre o Conjunto de Dados")
    st.warning(
        "O conjunto de dados contém informações sobre 891 passageiros do navio, incluindo:\n"
        "- Dados demográficos (idade, sexo)\n"
        "- Informações socioeconómicas (classe da passagem)\n"
        "- Detalhes da viagem (porto de embarque, cabine)\n"
        "- Estrutura familiar a bordo\n"
        "- Estado de sobrevivência\n"
        "- Tarifa paga pela passagem"
    )

    st.divider()
    st.markdown(
        """
        ### 🔗 Referências e Ligações Úteis
        - [Kaggle da Análise do Titanic](https://www.kaggle.com/code/joaobacalhau/notebooka528748a6b)  
        - [GitHub do Projeto](https://github.com/joaocba/data_analysis_titanic)
        - [Documentação das Bibliotecas Utilizadas](https://scikit-learn.org/)

        ### 👨‍💻 Desenvolvido por:
        - João C. Bacalhau

        ### 🎓 Sobre o Curso:
        - **Coordenadora/Formadora:** Dra. Ana Grade 
        - **Curso:** Projeto Final do Curso de Analista de Dados do Citeforma
        """
    )
    st.divider()

show()
