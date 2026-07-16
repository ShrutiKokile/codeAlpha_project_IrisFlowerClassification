import streamlit as st
import pickle
import numpy as np

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Page Settings
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Species Classification")
st.markdown("---")

st.write(
    "Enter flower measurements and the Machine Learning model "
    "will predict the flower species."
)

# Input Fields
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# Predict Button
if st.button("🔍 Predict Flower Species"):

    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    species = encoder.inverse_transform(
        prediction
    )[0]

    probability = model.predict_proba(sample)

    confidence = probability.max() * 100

    st.success(f"Predicted Species: {species}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    # Display image and description
    if "setosa" in species.lower():

        st.image("setosa.jpg", width=350)

        st.subheader("🌸 Iris Setosa")

        st.write(
            "Setosa flowers have short petals and are easy to identify."
        )

    elif "versicolor" in species.lower():

        st.image("versicolor.jpg", width=350)

        st.subheader("🌼 Iris Versicolor")

        st.write(
            "Versicolor flowers have medium-sized petals and balanced features."
        )

    else:

        st.image("virginica.jpg", width=350)

        st.subheader("🌺 Iris Virginica")

        st.write(
            "Virginica flowers have larger petals and wider dimensions."
        )

st.markdown("---")

st.header("📊 Project Information")

st.write("""
**Project Name:** Iris Flower Species Classification

**Goal:** Predict the species of an Iris flower using Machine Learning.

**Species:**
- Iris Setosa
- Iris Versicolor
- Iris Virginica

**Algorithm Used:** Random Forest Classifier

**Expected Accuracy:** 95% - 100%
""")

st.markdown("---")

st.write("Developed using Python, Scikit-Learn and Streamlit")