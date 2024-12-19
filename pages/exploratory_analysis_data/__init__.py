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

__all__ = [
    'age_analysis',
    'fare_analysis',
    'family_analysis',
    'gender_analysis',
    'general_survival',
    'age_survival',
    'gender_survival',
    'class_survival',
    'family_survival',
    'port_survival',
    'combined_survival',
    'correlation_analysis'
]