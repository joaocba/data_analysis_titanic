# Configurações da aplicação
APP_CONFIG = {
    'title': 'Análise do Titanic',
    'layout': 'wide',
    'theme': {
        'primaryColor': '#1e3a8a',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f8f9fa',
        'textColor': '#374151',
        'font': 'sans-serif'
    },
    'menu_items': {
        'About': 'Análise de sobrevivência do Titanic'
    }
}

# Configurações de visualização
PLOT_CONFIG = {
    'figure_size': (10, 6),
    'style': 'seaborn',
    'palette': ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd'],
    'background': '#ffffff',
    'grid_color': '#f3f4f6'
}

# Variáveis globais
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"