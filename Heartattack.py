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


# _____________________IMPORT DATASETS_____________________

df = pd.read_csv('/Users/toan/Arbeit/Ironhack/Miniproject/final_project/data/cleaned/df_ml_AgeGroup.csv')
df_ml = pd.read_csv('/Users/toan/Arbeit/Ironhack/Miniproject/final_project/data/cleaned/Medicaldataset_cleaned.csv')


# _____________________EDIT DATASET_____________________

cols = df.columns.tolist()
cols = ['AgeGroup'] + [col for col in cols if col != 'AgeGroup']
df = df[cols]
df = df.sort_values(by='AgeGroup', ascending=False)

df['Gender'] = df['Gender'].apply(lambda x: 'Male' if x == 1 else 'Female')

# _____________________MENU BAR_____________________


# _____________________INTRODUCTION_____________________

st.title("Myocardial Infarction")
st.write("")
st.video("Recognizing a heart Attack  3D animation Heart attack signs and symptoms.mp4")


# _____________________MAP_____________________


st.header("Zheen Hospital")

erbil_coords = pd.DataFrame({
    'lat': [36.1946894],
    'lon': [44.0443058]
})
st.map(erbil_coords)



# _____________________FILTER_____________________

st.header('Heart Attack')

#st.write('A heart attack (myocardial infarction) is a medical emergency where your heart muscle begins to die because it isn’t getting enough blood flow. A blockage in the arteries that supply blood to your heart usually causes this. If a healthcare provider doesn’t restore blood flow quickly, a heart attack can cause permanent heart damage and death.')

col1, col2 = st.columns([2,1])

with col1:
    st.write('A heart attack (myocardial infarction) is an extremely dangerous condition that happens because you don’t have enough blood flow to some of your heart muscle. This lack of blood flow can occur because of many different factors but is usually related to a blockage in one or more of your heart’s arteries.')
    st.write('')
    st.write('Without blood flow, the affected heart muscle will begin to die. If you don’t get blood flow back quickly, a heart attack can cause permanent heart damage and/or death.')
    st.write('A heart attack is a life-threatening emergency. If you think you or someone you’re with is having a heart attack, call 911 (or your local emergency services phone number). Time is critical in treating a heart attack. A delay of even a few minutes can result in permanent heart damage or death.')

with col2:
    st.image("HeartAttack.jpg")


st.write('')
st.write('')
st.markdown('### What exactly happens during a heart attack?')
st.write('When a heart attack happens, blood flow to a part of your heart stops or is far below normal, which causes injury or death to that part of your heart muscle. When a part of your heart can’t pump because it’s dying from lack of blood flow, it can disrupt the pumping function of your heart. This can reduce or stop blood flow to the rest of your body, which can be deadly if someone doesn’t correct it quickly.')

st.write('')
st.write('')

st.markdown('### Signs and Symptoms')

col1, col2 = st.columns([1,1])

with col1:
    st.markdown('### What does a heart attack feel like?')
    st.write('Many people feel pain in their chest during a heart attack. It can feel like discomfort, squeezing or heaviness, or it can feel like crushing pain. It may start in your chest and spread (or radiate) to other areas like your left arm (or both arms), shoulder, neck, jaw, back or down toward your waist.')
    st.write('')
    st.write('People often think they’re having indigestion or heartburn when they’re actually having a heart attack.')
    st.write('')
    st.write('Some people only experience shortness of breath, nausea or sweating.')

with col2:
    st.markdown('### What are the symptoms of a heart attack?')
    st.write('Heart attacks can have many symptoms, some of which are more common than others.')
    st.write('')
    st.write('Heart attack symptoms that people describe most often include:')
    st.markdown("""
    - Chest pain (angina)
    - Shortness of breath or trouble breathing
    - Trouble sleeping (insomnia)
    - Nausea or stomach discomfort
    - Heart palpitations
    - Anxiety or a feeling of “impending doom.”
    - Feeling lightheaded, dizzy or passing out.
    """)

st.write('')
st.write('')

col1, col2 = st.columns(2)

with col1:
    st.markdown('### What causes a heart attack?')
    st.write('Most heart attacks happen because of a blockage in one of the blood vessels that supply your heart. Most often, this occurs because of plaque, a sticky substance that can build up on the insides of your arteries (similar to how pouring grease down your kitchen sink can clog your home plumbing). That buildup is called atherosclerosis. When there’s a large amount of this atherosclerotic buildup in the blood vessels to your heart, this is called coronary artery disease.')
    st.write('')
    st.write('Sometimes, plaque deposits inside the coronary (heart) arteries can break open or rupture, and a blood clot can get stuck where the rupture happened. If the clot blocks the artery, this can deprive the heart muscle of blood and cause a heart attack.')
    st.write('')
    st.write('Heart attacks are possible without ruptured plaque, but this is rare and only accounts for about 5% of all heart attacks. This kind of heart attack can occur for the following reasons:')
    st.markdown("""
    - Coronary artery spasm
    - Rare medical conditions, like any disease that causes unusual narrowing of blood vessels
    - Trauma that causes tears or ruptures in your coronary arteries
    - Obstruction that came from somewhere else in your body, like a blood clot or air bubble (embolism) that ends up in a coronary artery
    - Eating disorders. Over time, these can damage your heart and ultimately result in a heart attack
    - Anomalous coronary arteries (a heart issue you’re born with where the coronary arteries are in abnormal positions. Compression of these causes a heart attack)
    - Other conditions that can cause your heart not to receive as much blood as it should for a prolonged period of time, such as when blood pressure is too low, oxygen is too low or your heart rate is too fast
    """)

with col2:
    st.markdown('### What are the risk factors for a heart attack?')
    st.write('Several key factors affect your risk of having a heart attack. Unfortunately, some of these heart attack risk factors aren’t things you can modify:')
    st.write('')
    st.markdown("""
    - Age and sex: Your risk of heart attack increases as you get older. Your sex influences when your risk of a heart attack starts to increase. For people assigned male at birth (AMAB), the risk of heart attack increases at age 45. For people assigned female at birth (AFAB), the risk of heart attack increases at age 50 or after menopause
    - Family history of heart disease: If you have a parent or sibling with a history of heart disease or heart attack — especially at a younger age — your risk is even greater because your genetics are similar to theirs. Your risk increases if a first-degree relative (biological sibling or parent) received a heart disease diagnosis at age 55 or younger if they’re AMAB, or at age 65 or younger if they’re AFAB
    - Lifestyle: Lifestyle choices you make that aren’t good for your heart can increase your risk of having a heart attack. This includes things like smoking, eating high-fat foods, lack of physical activity, drinking too much alcohol and drug use
    - Certain health conditions: Some health conditions put stress on your heart and increase your risk for heart attack. This includes diabetes, obesity, high blood pressure, high cholesterol, eating disorders or a history of preeclampsia
    """)


# _____________________FILTER_____________________


st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 150px; /* Adjusted width for the sidebar */
    }
    .sidebar .multiselect-container {
        width: 120px !important; /* Adjusted width for multiselect options */
    }
    .sidebar .stRadio {
        padding: 0px 2px; /* Adjusted padding for radio buttons */
    }
    .sidebar .stButton > button {
        padding: 0.25rem 0.5rem; /* Adjusted padding for buttons */
        font-size: 0.8rem; /* Adjusted font size for buttons */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create Filter Sidebar

st.sidebar.header("Please Filter Here:")
result = st.sidebar.multiselect(
    "Select Result:",
    options=df["Result"].unique(),
    default=df["Result"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

agegroup = st.sidebar.multiselect(
    "Select AgeGroup:",
    options=df["AgeGroup"].unique(),
    default=df["AgeGroup"].unique()
)


df_selection = df.query(
    "Gender == @gender & AgeGroup == @agegroup & Result == @result"
)



# _____________________STATISTICS AND PLOTTINGS_____________________

st.header("Statistics")

positive_results = df_selection[df_selection['Result'] == 'positive']

agegroup_counts_positive = positive_results['AgeGroup'].value_counts().sort_index()

total_counts = df_selection['AgeGroup'].value_counts().sort_index()
mean_results = agegroup_counts_positive / total_counts

fig1 = px.bar(x=agegroup_counts_positive.index, y=agegroup_counts_positive.values,
              labels={'x': 'Age', 'y': 'Total Heart Attacks'},
              title='Myocardial Infarction by Age')

fig2 = px.bar(x=mean_results.index, y=mean_results.values,
              labels={'x': 'Age', 'y': 'Myocardial Infarction (Relative)'})
              #title='Myocardial Infarction (Relative) by Age')

# Streamlit layout

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1)

with col2:
    st.plotly_chart(fig2)


#________Troponin and CK-MB________

df_selection = df_selection.sort_values('AgeGroup')

fig_troponin = px.bar(df_selection, x='AgeGroup', y='Troponin',
                      title='Troponin Levels by Age Group',
                      labels={'AgeGroup': 'Age Group', 'Troponin': 'Troponin Level'})

fig_troponin.update_layout(xaxis_title='Age Group',
                           yaxis_title='Troponin Level')

fig_ckmb = px.bar(df_selection, x='AgeGroup', y='CK-MB',
                  title='CK-MB Levels by Age Group',
                  labels={'AgeGroup': 'Age Group', 'CK-MB': 'CK-MB Level'})

fig_ckmb.update_layout(xaxis_title='Age Group',
                       yaxis_title='CK-MB Level')



col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_troponin)

with col2:
    st.plotly_chart(fig_ckmb)


#________Distribution of Result and Gender________

fig_result_box = px.box(df_selection, x='Result', y=df_selection.groupby('Result').cumcount(),
                        title='Distribution of Myocardial Infarction',
                        labels={'Result': 'Result', 'y': 'Count'})

fig_gender_box = px.box(df_selection, x='Gender', y=df_selection.groupby('Gender').cumcount(),
                        #title='Distribution of Gender',
                        labels={'Gender': 'Gender', 'y': 'Count'})

result_counts = df_selection['Result'].value_counts()

gender_counts = df_selection['Gender'].value_counts()

fig_result_pie = px.pie(values=result_counts.values, names=result_counts.index,
                        #title='Distribution of Myocardial Infarction Results',
                        labels={'names': 'Result', 'values': 'Count'})

fig_gender_pie = px.pie(values=gender_counts.values, names=gender_counts.index,
                        title='Distribution of Gender',
                        labels={'names': 'Gender', 'values': 'Count'})

#st.subheader('Distribution of Myocardial Infarction and Gender')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_result_box)

with col2:
    st.plotly_chart(fig_result_pie)

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(fig_gender_box)

with col4:
    st.plotly_chart(fig_gender_pie)




# _____________________MACHINE LEARNING MODEL_____________________

st.image("Heartbeat.png")
st.header("Prediction of Myocardial Infarction")
st.write("")
st.write("After entering the parameters, the selected machine learning model calculates the probability of a heart attack.")


st.sidebar.subheader('Machine Learning Model')

ml_model = st.sidebar.selectbox(
    "Select Model:",
    options=['Random Forest', 'Gradient Boosting']
)

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

# Collect user input for prediction
col1, col2 = st.columns(2)
with col1:
    gender = st.radio('Gender', ['Male', 'Female'])
with col2:
    age = st.slider('Age', 18, 80, int(df_ml['Age'].mean()))
col3, col4 = st.columns(2)
with col3:
    heart_rate = st.slider('Heart Rate', 30, 220, int(df_ml['Heart_rate'].mean()))
with col4:
    blood_sugar = st.slider('Blood Sugar', 50, 300, int(df_ml['Blood_sugar'].mean()))

col5, col6 = st.columns(2)
with col5:
    systolic_bp = st.slider('Systolic Blood Pressure', 60, 200, int(df_ml['Systolic_blood_pressure'].mean()))
with col6:
    diastolic_bp = st.slider('Diastolic Blood Pressure', 30, 120, int(df_ml['Diastolic_blood_pressure'].mean()))

ck_mb = st.slider('CK-MB', 0.000, 100.000, float(df_ml['CK-MB'].mean()), step=0.001)
troponin = st.slider('Troponin', 0.00, 10.00, float(df_ml['Troponin'].mean()), step=0.01)

# Prepare data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Male' else 0],
    'Heart_rate': [heart_rate],
    'Systolic_blood_pressure': [systolic_bp],
    'Diastolic_blood_pressure': [diastolic_bp],
    'Blood_sugar': [blood_sugar],
    'CK-MB': [ck_mb],
    'Troponin': [troponin]
})

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Display prediction result
#st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
st.write(f"Probability (Positive): {prediction_proba[0][1]:.2f}")



# ___________________PREVENTION____________________


st.markdown('### Can a heart attack be prevented? ###')
st.write('In general, there are many things that you can do that may prevent a heart attack. However, there are some factors you can’t change — especially your family history — that can still lead to a heart attack despite your best efforts. Still, reducing your risk can postpone when you have a heart attack and reduce the severity if you have one..')
st.write('')
st.write('**How can I lower my risk?**')
st.write('')
st.write('Although several risk factors can’t be modified, there are many ways you can help yourself and reduce your risk of a heart attack. These include:')
st.markdown("""
- Chest pain (angina)
- Shortness of breath or trouble breathing
- Trouble sleeping (insomnia)
- Nausea or stomach discomfort
- Heart palpitations
- Anxiety or a feeling of “impending doom.”
- Feeling lightheaded, dizzy or passing out.
    """)


# __________________________Literature_________________

st.subheader('More Informations')

st.write("Websites:")

st.write("https://www.heart.org/ American Heart Association (AHA)")
st.write("https://www.dhzc.charite.de/ratgeber/herzinfarkt/ Deutsches Herzzentrum der Charité (DHZC)")

st.write("Papers:")
st.write("[Oxford](https://www.ox.ac.uk/news/2023-11-13-ai-tool-could-help-thousands-avoid-fatal-heart-attacks) AI tool could help thousands avoid fatal heart attacks")
st.write("[Nature Science](https://www.nature.com/articles/s41591-023-02325-4) Machine learning for diagnosis of myocardial infarction using cardiac troponin concentrations")
