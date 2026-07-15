import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


@st.cache_resource
def load_and_train():
    data = pd.read_csv("StudentsPerformance.csv")

    X = data[['gender', 'race/ethnicity', 'parental level of education',
              'lunch', 'test preparation course', 'reading score', 'writing score']]
    y = data['math score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education',
                        'lunch', 'test preparation course']
    numerical_cols = ['reading score', 'writing score']

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, predictions), 2)
    r2 = round(r2_score(y_test, predictions), 4)

    return model, data, mae, r2


model, data, mae, r2 = load_and_train()

st.title("Student Math Score Predictor")
st.write("Predicts a student's math score from demographic and academic performance factors.")
st.caption(f"Model performance on held-out test data — MAE: {mae}, R²: {r2}")

st.divider()

gender = st.selectbox("Gender", sorted(data['gender'].unique()))
ethnicity = st.selectbox("Race/Ethnicity", sorted(data['race/ethnicity'].unique()))
education = st.selectbox("Parental Level of Education", sorted(data['parental level of education'].unique()))
lunch = st.selectbox("Lunch Type", sorted(data['lunch'].unique()))
prep = st.selectbox("Test Preparation Course", sorted(data['test preparation course'].unique()))
reading = st.slider("Reading Score", 0, 100, 70)
writing = st.slider("Writing Score", 0, 100, 70)

if st.button("Predict Math Score"):
    input_df = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': ethnicity,
        'parental level of education': education,
        'lunch': lunch,
        'test preparation course': prep,
        'reading score': reading,
        'writing score': writing
    }])
    prediction = model.predict(input_df)[0]
    st.metric("Predicted Math Score", round(prediction, 1))
