# Lung Cancer Risk Analysis & Dashboard

This project consists of two principal components designed to facilitate a systematic investigation and applied modeling of lung cancer risk:

1. **Cancer Risk.ipynb** — A Jupyter notebook that conducts comprehensive exploratory data analysis, data preprocessing, and preliminary statistical modeling. The notebook evaluates the predictive relevance of key variables (Age, Smokes, AreaQ, Alkhol, Result), implements foundational machine-learning procedures, and establishes the methodological basis for subsequent model deployment.  
2. **lung_cancer_dashboard.py** — A Streamlit application that operationalizes the analytical workflow into an interactive decision-support dashboard. The interface provides visualization modules, model performance summaries, and a prediction mechanism enabling end-users to estimate individual lung cancer risk based on supplied input characteristics.

## Dashboard Features
- Analytical interface modeled after formal decision-support and business-intelligence systems  
- Adjustable data-filtering capabilities (age strata, smoking status, environmental exposure, alcohol consumption)  
- Receiver Operating Characteristic (ROC) curve comparisons for model discrimination assessment  
- Confusion matrix visualizations for evaluating classification outcomes  
- Feature importance profiles to contextualize variable contributions  
- Key Performance Indicators (AUC, accuracy, precision, recall) summarized for model interpretability  
- A structured prediction module for generating patient-specific risk estimations

## How to Run
Install the necessary dependencies:

pip install streamlit pandas numpy scikit-learn plotly joblib

Execute the dashboard:

streamlit run lung_cancer_dashboard.py

Following execution, the application becomes accessible at:

http://localhost:8501

## Purpose
This project is designed to illustrate a complete analytical pipeline conforming to applied research standards: beginning with exploratory and statistical evaluation in a controlled notebook environment, progressing through machine-learning model development and validation, and culminating in an interactive deployment framework suitable for risk assessment, demonstration, or further methodological extension.

## Author
Mayowa Oluyole
