import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

# --- Function to add your custom background video ---
def add_video_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none; /* Hide the default background */
        }}

        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1; /* Send it to the back */
        }}
        </style>
        
        <video autoplay loop muted playsinline id="myVideo">
          <source src="https://github.com/Shivm-02/iris-app/raw/refs/heads/main/dancing%20lights.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# --- Call the function to set the background ---
add_video_bg()

# --- Model Training (cached so it only runs once) ---
@st.cache_data
def train_model():
    iris_dataset = load_iris()
    X = iris_dataset.data
    y = iris_dataset.target
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model, iris_dataset

model, iris_dataset = train_model()

# --- Load Data into a Pandas DataFrame for Charting ---
iris_df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
iris_df['species'] = [iris_dataset.target_names[i] for i in iris_dataset.target]

# --- App Sidebar ---
st.sidebar.header("About the App")
st.sidebar.info("This app predicts Iris flower species using a KNN model and visualizes the dataset.")
st.sidebar.markdown("[View the Iris Dataset on Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)")

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
        predicted_species_name = iris_dataset.target_names[prediction[0]]
        
        st.success(f"Predicted Species: **{predicted_species_name.upper()}**")
        st.balloons()
        
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
