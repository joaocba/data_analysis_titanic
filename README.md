# 🚢 Titanic Project - Survival Analysis

## 📋 About the Project

This project consists of an interactive web application developed using Streamlit to analyze the Titanic passenger data. The main goal is to identify survival patterns and develop predictive models, enabling a better understanding of which passenger characteristics influenced their chances of survival during the disaster.

## 🌐 Live Preview of the Application
[Streamlit App](https://joaocba-titanic-analysis.streamlit.app/)

### 🎯 Main Features

- Interactive exploratory data analysis
- Detailed visualizations by gender, class, and family structure
- Predictive modeling using Decision Tree and KNN
- Correlation analysis between variables
- Model performance comparison

## 🛠️ Requirements and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/titanic-project.git
cd titanic-project
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Main Dependencies
```
streamlit==1.28.2
pandas==2.1.2
numpy==1.26.1
matplotlib==3.8.1
seaborn==0.13.0
scikit-learn==1.3.2
```

## 🚀 How to Run

1. Activate the virtual environment (if you're using one):
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and access `http://localhost:8501`

## 📁 Project Structure

```
project/
├── pages/
│   ├── exploratory_analysis_data/   # Subfolder for detailed data exploration
│   │   ├── distributions/           # Analysis of variable distributions
│   │   │   ├── age_analysis.py      # Analyzes age distribution
│   │   │   ├── fare_analysis.py     # Analyzes fare distribution
│   │   │   ├── family_analysis.py   # Analyzes family size and structure
│   │   │   └── gender_analysis.py   # Analyzes gender distribution
│   │   ├── survival/                # Subfolder for survival analysis
│   │   │   ├── age_survival.py      # Analyzes survival rates by age groups
│   │   │   ├── class_survival.py    # Analyzes survival rates by passenger class
│   │   │   ├── combined_survival.py # Combines multiple survival factors
│   │   │   ├── family_survival.py   # Analyzes survival based on family structure
│   │   │   ├── gender_survival.py   # Analyzes survival rates by gender
│   │   │   ├── general_survival.py  # Provides general survival statistics
│   │   │   └── port_survival.py     # Analyzes survival based on port of embarkation
│   │   └── correlation_analysis.py  # Analyzes correlations between variables
│   ├── data_cleaning.py             # Data cleaning and preprocessing steps
│   ├── initial_analysis.py          # Initial overview of the dataset
│   ├── intro.py                     # Introduction and project objectives
│   ├── modeling.py                  # Implementation of predictive models
│   └── exploratory_analysis.py      # General exploratory analysis script
├── utils/
│   ├── data_loader.py               # Functions to load datasets
│   ├── data_processor.py            # Functions for data transformation
│   └── visualization.py             # Visualization utilities
├── streamlit_app.py                 # Main application script
└── config.py                        # Configuration file for global settings
```

## 📊 Dataset

The dataset contains information about 891 Titanic passengers, including:
- Demographic data (age, gender)
- Socioeconomic information (ticket class)
- Trip details (port of embarkation, cabin)
- Family structure aboard
- Survival status
- Ticket fare

## 🔍 Methodology

### Data Preparation
- Handling missing values
- Encoding categorical variables
- Creating new features
- Cleaning and validating data
- Normalizing numerical variables

### Exploratory Analysis
- Studying variable distributions
- Identifying patterns and correlations
- Survival analysis by groups
- Visualizing key relationships
- Relevant statistical tests

### Modeling
- Implementing different algorithms
- Performance evaluation and comparison
- Selecting the best model
- Hyperparameter optimization
- Cross-validating results

## 👨‍💻 Development

### Author
- João C. Bacalhau

### Course
- **Instructor:** Dr. Ana Grade 
- **Course:** Final Project of the Data Analyst Course at Citeforma

## 🔗 Useful Links
- [Kaggle Titanic Analysis](https://www.kaggle.com/code/joaobacalhau/notebooka528748a6b)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📜 License

This project is licensed under the [MIT License](LICENSE).
