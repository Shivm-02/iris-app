import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

# --- Function to add a background video and all UI styling ---
def add_bg_and_styling():
    st.markdown(
        f"""
        <style>
        /* --- Google Font Import --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        /* --- Video Background --- */
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100vw;
            min-height: 100vh;
            width: auto;
            height: auto;
            object-fit: cover;
            z-index: -1000;
        }}

        /* --- Main App Styling --- */
        .stApp {{
            background: transparent;
            font-family: 'Inter', sans-serif;
        }}

        /* --- Glassmorphism Containers --- */
        .main, [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(0, 0, 0, 0.55); /* Increased darkness for better contrast */
            backdrop-filter: blur(10px); /* The frosted glass effect */
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }}

        /* --- Text Color & Shadow for Readability --- */
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp .stMarkdown, .stApp label {{
            color: #FFFFFF; /* White text */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Added text shadow for pop */
        }}

        /* --- Seamless Image Fade-in Animation --- */
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: scale(0.95); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}

        .fade-in-image {{
            animation: fadeIn 0.8s ease-in-out;
            border-radius: 10px; /* Rounded corners for the image */
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); /* Subtle shadow */
        }}
        </style>
        
        <video autoplay loop muted playsinline id="myVideo">
          <source src="https://github.com/Shivm-02/iris-app/raw/refs/heads/main/dg%20bg%20final.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# --- Call the function to set the background and styles ---
add_bg_and_styling()

# --- Model Training ---
@st.cache_data
def train_model():
    iris_dataset = load_iris()
    X = iris_dataset.data
    y = iris_dataset.target
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model, iris_dataset

model, iris_dataset = train_model()

# --- Load Data into a Pandas DataFrame ---
iris_df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
iris_df['species'] = [iris_dataset.target_names[i] for i in iris_dataset.target]

# --- App Sidebar ---
st.sidebar.header("About the App")
st.sidebar.info("This app predicts Iris flower species using a KNN model and visualizes the dataset.")

# --- Main App Interface ---
st.title("Iris Flower Species Predictor")

tab1, tab2 = st.tabs(["📊 Predict Species", "📈 Data Explorer"])

with tab1:
    st.header("Enter Flower Measurements")
    col1, col2 = st.columns(2)
    with col1:
        sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
        petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 3.4)
    with col2:
        sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.2)
        petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    if st.button("Predict", type="primary"):
        new_flower = [[sepal_l, sepal_w, petal_l, petal_w]]
        prediction = model.predict(new_flower)
        prediction_proba = model.predict_proba(new_flower)
        predicted_species_name = iris_dataset.target_names[prediction[0]]
        
        st.success(f"Predicted Species: **{predicted_species_name.upper()}**")
        
        confidence = np.max(prediction_proba) * 100
        st.info(f"Confidence Score: {confidence:.2f}%")
        
        # --- Using Markdown for Animated Image ---
        image_url = ""
        if predicted_species_name == 'setosa':
            image_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg"
        elif predicted_species_name == 'versicolor':
            image_url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
        else:
            image_url = "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
        
        # Display the image with the fade-in class and constrained size
        st.markdown(f'<img src="{image_url}" class="fade-in-image" style="max-width: 500px; display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)

        st.session_state['new_flower_data'] = {
            'sepal length (cm)': sepal_l, 'sepal width (cm)': sepal_w,
            'petal length (cm)': petal_l, 'petal width (cm)': petal_w,
            'species': 'Your Flower'
        }

with tab2:
    st.header("Interactive Data Chart")
    chart_data = iris_df.copy()
    
    if 'new_flower_data' in st.session_state:
        new_point = pd.DataFrame([st.session_state['new_flower_data']])
        chart_data = pd.concat([chart_data, new_point], ignore_index=True)

    st.scatter_chart(chart_data, x='sepal length (cm)', y='sepal width (cm)', color='species', size='petal length (cm)')

