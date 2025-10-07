import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# --- Model Training ---
# Load the dataset and train the model (this happens once when the app starts)
iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# --- App Sidebar ---
st.sidebar.header("About the App")
st.sidebar.info("This web app was created to predict the species of an Iris flower based on its physical measurements. It uses a K-Nearest Neighbors (KNN) model trained on the classic Iris dataset.")
st.sidebar.markdown("[View the Iris Dataset on Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)")

# --- Main Web App Interface ---
st.title("Iris Flower Species Predictor 🌸")
st.markdown("Enter the flower measurements below to get a prediction.")

# --- Input Fields in Columns for a Cleaner Look ---
col1, col2 = st.columns(2)
with col1:
    sepal_l = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
    petal_l = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
with col2:
    sepal_w = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_w = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# --- Prediction Logic ---
if st.button("Predict Species", type="primary"):
    # Prepare the input for the model
    new_flower = [[sepal_l, sepal_w, petal_l, petal_w]]
    
    # Make the prediction and get probabilities
    prediction = model.predict(new_flower)
    prediction_proba = model.predict_proba(new_flower)
    
    # Get the name for the predicted species
    predicted_species_name = iris_dataset.target_names[prediction[0]]
    
    # --- Display the Results ---
    st.success(f"The predicted species is: **{predicted_species_name.upper()}**")
    
    # Display the confidence score
    confidence = np.max(prediction_proba) * 100
    st.info(f"Confidence Score: {confidence:.2f}%")
    
    # Display an image of the flower
    if predicted_species_name == 'setosa':
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa")
    elif predicted_species_name == 'versicolor':
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica")