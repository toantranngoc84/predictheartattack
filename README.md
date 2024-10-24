# Heart Attack Prediction


## Overview

A heart attack, also known as myocardial infarction, is a medical emergency where the heart muscle begins to die because it isn’t receiving enough blood flow. This is typically caused by blockages in the arteries that supply blood to the heart. Without prompt medical intervention to restore blood flow, a heart attack can lead to permanent damage or death.

This Streamlit-based application provides crucial information about heart attacks and offers a predictive tool using a machine learning model to assess the risk of heart attacks based on user inputs.


## Features

### 1. **Informational Content**
The app includes detailed educational content on heart attacks:
- **What is a Heart Attack?**
- **Signs and Symptoms**
- **What Causes a Heart Attack?**
- **Risk Factors**
- **Prevention**
- **Statistics**
- **Further Information** (Helpful websites, research papers, etc.)

### 2. **Heart Attack Prediction Tool**
The app allows users to input clinical parameters to predict the likelihood of a heart attack using a trained machine learning model. Input parameters include:
- Gender
- Age
- Heart Rate
- Blood Sugar Level
- Systolic Blood Pressure
- Diastolic Blood Pressure
- CK-MB (Creatine Kinase-MB) Level
- Troponin Level

The model then provides a prediction of the likelihood of a heart attack based on these inputs.


## Dataset

The dataset used to train the predictive model is sourced from [Mendeley](https://data.mendeley.com/datasets/wmhctcrt5v/1).


## Machine Learning Approach

The heart attack prediction model was built using Python, employing various machine learning techniques:

- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to handle class imbalance in the dataset.
- **StandardScaler**: Applied for feature normalization.
- **Hyperparameter Tuning**: Optimized model performance through techniques like GridSearchCV and RandomizedSearchCV.
- **Cross-Validation**: Used to evaluate model performance with K-Fold cross-validation.
- **Feature Selection**: Multiple feature selection techniques were employed.
- **Model Selection**: Several classification models were explored, including boosting models (e.g., XGBoost, LightGBM), Random Forest, and Logistic Regression. The best-performing model is deployed in the app.


## How to Use the App

The deployed app is available online for immediate use:

🔗 **[Predict Heart Attack Streamlit App](https://predictheartattack.streamlit.app/)**

### Steps to Predict Heart Attack Risk:
1. Visit the app via the link above.
2. Navigate to the "Prediction of Myocardial Infarction" section.
3. Enter the required parameters (e.g., gender, age, heart rate, etc.).
4. The app will calculate and display the likelihood of a heart attack.



## Installation

To run the app locally on your machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/toantranngoc84/Myocardial_Infarction.git

2. **Navigate into the project directory:**:
   ```bash
   cd Myocardial_Infarction

3. **Install the required dependencies: Install the dependencies listed in the requirements.txt file:**:
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit app: After installing the dependencies, you can start the app by running:**:
   ```bash
   streamlit run Heartattack.py

5. **Access the app: Open your browser and go to http://localhost:8501 to interact with the app locally.**:


## Requirements

Below are the main Python packages required to run this project:

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn

You can install these dependencies automatically using the provided `requirements.txt` file.

