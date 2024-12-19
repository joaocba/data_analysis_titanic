import matplotlib.pyplot as plt
import numpy as np

# Configurações globais de visualização
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

# Definição global das cores
COLORS = {
    'primary': plt.cm.tab20(0),  # Azul
    'secondary': plt.cm.tab20(2),  # Verde
    'negative': plt.cm.tab20(3),  # Vermelho
    'neutral': plt.cm.tab20(4),  # Roxo
    'background': '#f0f2f6',
    'text': '#262730',
}


def get_color_palette(n_colors):
    """Retorna uma paleta de cores consistente
    """
    return plt.cm.tab20(np.linspace(0, 1, n_colors))


def set_plot_style(ax, title, xlabel, ylabel):
    """Aplica estilo consistente ao plot
    """
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)


def format_currency(value):
    """Formata valor para moeda (£)
    """
    return f"£{value:.2f}"


def format_percentage(value):
    """Formata valor para percentagem
    """
    return f"{value:.1f}%"