# _____________________LIBRARIES_____________________

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import geopandas as gpd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import joblib

# _____________________IMPORT DATASETS_____________________

df = pd.read_csv('/Users/toan/Arbeit/Ironhack/Miniproject/final_project/data/cleaned/df_ml_AgeGroup.csv')
df_ml = pd.read_csv('/Users/toan/Arbeit/Ironhack/Miniproject/final_project/data/cleaned/Medicaldataset_cleaned.csv')


# _____________________EDIT DATASET_____________________

# Reorder columns (put 'Agegroup' first)
cols = df.columns.tolist()
cols = ['AgeGroup'] + [col for col in cols if col != 'AgeGroup']
df = df[cols]
df = df.sort_values(by='AgeGroup', ascending=False)

# Map Gender to Male for 1, Female for 0
df['Gender'] = df['Gender'].apply(lambda x: 'Male' if x == 1 else 'Female')

# _____________________MENU BAR_____________________




# _____________________INTRODUCTION_____________________

st.title("Myocardial Infarction")
st.write("")
st.video("Recognizing a heart Attack  3D animation Heart attack signs and symptoms.mp4")


# _____________________MAP_____________________

st.header("Zheen Hospital")
# Coordinates for Erbil, Iraq
erbil_coords = pd.DataFrame({
    'lat': [36.1946894],
    'lon': [44.0443058]
})

st.map(erbil_coords)


# _____________________FILTER_____________________

st.header("Statistics")

# Sidebar width customization
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 180px;
    }
    .sidebar .multiselect-container {
        width: 150px !important;
    }
    .sidebar .stRadio {
        padding: 0px 3px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Create Filter Sidebar

# Select Result
st.sidebar.header("Please Filter Here:")
result = st.sidebar.multiselect(
    "Select Result:",
    options=df["Result"].unique(),
    default=df["Result"].unique()
)

# Select Gender
gender = st.sidebar.multiselect(
    "Select Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

# Select Age Group
agegroup = st.sidebar.multiselect(
    "Select AgeGroup:",
    options=df["AgeGroup"].unique(),
    default=df["AgeGroup"].unique()
)


df_selection = df.query(
    "Gender == @gender & AgeGroup == @agegroup & Result == @result"
)

# Dispaly Dataframe
#st.dataframe(df_selection)


# _____________________STATISTICS AND PLOTTINGS_____________________

# Filter the dataset for positive results
positive_results = df_selection[df_selection['Result'] == 'positive']

# Count the occurrences of each AgeGroup for positive results
agegroup_counts_positive = positive_results['AgeGroup'].value_counts().sort_index()

# Compute relative frequencies (mean results)
total_counts = df_selection['AgeGroup'].value_counts().sort_index()
mean_results = agegroup_counts_positive / total_counts

# Create bar plot for Number of Heart Attacks by Age
fig1 = px.bar(x=agegroup_counts_positive.index, y=agegroup_counts_positive.values,
              labels={'x': 'Age', 'y': 'Total Heart Attacks'},
              title='Myocardial Infarction by Age')

# Create bar plot for Myocardial Infarction (Relative) by Age
fig2 = px.bar(x=mean_results.index, y=mean_results.values,
              labels={'x': 'Age', 'y': 'Myocardial Infarction (Relative)'})
              #title='Myocardial Infarction (Relative) by Age')

# Streamlit layout
#st.subheader('Title')

# Display the plots side by side
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1)

with col2:
    st.plotly_chart(fig2)


#________Troponin and CK-MB________

# Sort the DataFrame by AgeGroup
df_selection = df_selection.sort_values('AgeGroup')

# Create the bar plot for Troponin
fig_troponin = px.bar(df_selection, x='AgeGroup', y='Troponin', 
                      title='Troponin Levels by Age Group',
                      labels={'AgeGroup': 'Age Group', 'Troponin': 'Troponin Level'})

# Customize layout (optional)
fig_troponin.update_layout(xaxis_title='Age Group',
                           yaxis_title='Troponin Level')

# Create the bar plot for CK-MB
fig_ckmb = px.bar(df_selection, x='AgeGroup', y='CK-MB', 
                  title='CK-MB Levels by Age Group',
                  labels={'AgeGroup': 'Age Group', 'CK-MB': 'CK-MB Level'})

# Customize layout (optional)
fig_ckmb.update_layout(xaxis_title='Age Group',
                       yaxis_title='CK-MB Level')


# Streamlit layout
#st.title('Troponin and CK-MB Levels by Age Group')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_troponin)

with col2:
    st.plotly_chart(fig_ckmb)


#________Distribution of Result and Gender________

# Create boxplot for Result vs Count
fig_result_box = px.box(df_selection, x='Result', y=df_selection.groupby('Result').cumcount(), 
                        title='Distribution of Myocardial Infarction',
                        labels={'Result': 'Result', 'y': 'Count'})

# Create boxplot for Gender vs Count
fig_gender_box = px.box(df_selection, x='Gender', y=df_selection.groupby('Gender').cumcount(), 
                        #title='Distribution of Gender',
                        labels={'Gender': 'Gender', 'y': 'Count'})

# Calculate the distribution of Result
result_counts = df_selection['Result'].value_counts()

# Calculate the distribution of Gender
gender_counts = df_selection['Gender'].value_counts()

# Create pie chart for Result distribution
fig_result_pie = px.pie(values=result_counts.values, names=result_counts.index,
                        #title='Distribution of Myocardial Infarction Results',
                        labels={'names': 'Result', 'values': 'Count'})

# Create pie chart for Gender distribution
fig_gender_pie = px.pie(values=gender_counts.values, names=gender_counts.index,
                        title='Distribution of Gender',
                        labels={'names': 'Gender', 'values': 'Count'})

# Streamlit layout
#st.subheader('Distribution of Myocardial Infarction and Gender')

# Boxplot and pie chart of Result side by side
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_result_box)

with col2:
    st.plotly_chart(fig_result_pie)

# Boxplot and pie chart of Gender side by side
col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(fig_gender_box)

with col4:
    st.plotly_chart(fig_gender_pie)



def radar_chart(df):
    # Selecting columns for the radar chart (excluding non-numeric columns like 'Gender')
    radar_cols = ['Age', 'Heart_rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Blood_sugar', 'CK-MB', 'Troponin']

    # Calculate the mean values for each column to be used as the radar chart's data
    radar_data = df[radar_cols].mean()

    # Create a list of column names for the radar chart
    categories = radar_data.index.tolist()

    # Create a list of values for each category
    values = radar_data.values.tolist()

    # Create a trace using go.Scatterpolar
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    # Updating the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df[radar_cols].max().max()]  # Adjust range as needed
            )),
        showlegend=True,
        #title='Radar Chart Example'
    )

    return fig

# Streamlit app
def main():
    #st.title('Radar Chart Example')

    st.subheader('Radar Chart')

    radar_fig = radar_chart(df_selection)
    st.plotly_chart(radar_fig)

if __name__ == '__main__':
    main()




# _____________________Machine Learning Model_____________________

# _____________________MACHINE LEARNING MODEL_____________________

# Sidebar section for selecting ML model
st.sidebar.subheader('Machine Learning Model')

# Select ML model
ml_model = st.sidebar.selectbox(
    "Select Model:",
    options=['Random Forest', 'Gradient Boosting']
)

# Train/test split and model fitting
X = df_ml.drop('Result', axis=1)
y = df_ml['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if ml_model == 'Random Forest':
    model = RandomForestClassifier()
elif ml_model == 'Gradient Boosting':
    model = GradientBoostingClassifier()

# Fit the model
model.fit(X_train, y_train)

# Evaluate model performance
accuracy = model.score(X_test, y_test)
st.sidebar.write(f'Model Accuracy: {accuracy:.2f}')

# _____________________PROBABILITY/FORECAST_____________________

# Prediction section
st.header("Prediction")

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_ml[['Age', 'Heart_rate', 'Systolic_blood_pressure', 
                                              'Diastolic_blood_pressure', 'Blood_sugar', 'CK-MB']])

# Create the dataframe with scaled features
df_feat = pd.DataFrame(scaled_features, columns=['Age', 'Heart_rate', 'Systolic_blood_pressure', 
                                                 'Diastolic_blood_pressure', 'Blood_sugar', 'CK-MB'])

# Define X and y
X = df_feat
y = df_ml['Result']  # Assuming 'Result' is your target variable

# Train the Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(logmodel, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the model and scaler
logmodel = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to display predictions
def display_predictions(age, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb):
    input_data = scaler.transform([[age, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb]])
    prediction = logmodel.predict(input_data)[0]

    st.subheader('Heart Attack Prediction')
    if prediction == 0:
        st.write("<span class='diagnosis bright-green'>Negative</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Positive</span>", unsafe_allow_html=True)

# Streamlit app
def main():
    st.title('Heart Attack Prediction App')

    # Add sliders or input components for each feature
    age = st.slider('Age', int(df_ml['Age'].min()), int(df_ml['Age'].max()), int(df_ml['Age'].mean()))
    heart_rate = st.slider('Heart Rate', int(df_ml['Heart_rate'].min()), int(df_ml['Heart_rate'].max()), int(df_ml['Heart_rate'].mean()))
    sys_bp = st.slider('Systolic Blood Pressure', int(df_ml['Systolic_blood_pressure'].min()), int(df_ml['Systolic_blood_pressure'].max()), int(df_ml['Systolic_blood_pressure'].mean()))
    dia_bp = st.slider('Diastolic Blood Pressure', int(df_ml['Diastolic_blood_pressure'].min()), int(df_ml['Diastolic_blood_pressure'].max()), int(df_ml['Diastolic_blood_pressure'].mean()))
    blood_sugar = st.slider('Blood Sugar', int(df_ml['Blood_sugar'].min()), int(df_ml['Blood_sugar'].max()), int(df_ml['Blood_sugar'].mean()))
    ck_mb = st.slider('CK-MB', float(df_ml['CK-MB'].min()), float(df_ml['CK-MB'].max()), float(df_ml['CK-MB'].mean()))

    if st.button('Predict'):
        display_predictions(age, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb)

if __name__ == '__main__':
    main()