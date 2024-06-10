import streamlit as st
import joblib
import numpy as np


def load_model():
    
    data = joblib.load(open("saved_steps.pkl", "rb"))
    return data

data = load_model()

regressor = data["model"]
preprocessor = data["preprocessor"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education_levels = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        X_transformed = preprocessor.transform(X)
        salary = regressor.predict(X_transformed)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f}")

# To run the Streamlit app, include this line at the end of your script
if __name__ == "__main__":
    show_predict_page()
