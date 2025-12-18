import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wine Cluster Analysis", page_icon="🍷", layout="wide")

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.markdown("""
    <style>
    .main { background-color: #fdfafb; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #722f37; /* Wine Red */
        color: white;
        height: 3em;
    }
    .cluster-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: #722f37;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    h1 { color: #722f37; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('wine_kmeans_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# --- HEADER ---
st.title("🍷 Wine Quality Cluster Predictor")
st.markdown("Enter the chemical properties of the wine to determine its category (Cluster).")
st.divider()

if model is None:
    st.error("Model file not found! Please ensure 'wine_kmeans_model.pkl' is in the same folder.")
    st.stop()

# --- INPUT FORM ---
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alcohol = st.number_input("Alcohol", value=13.0)
        malic_acid = st.number_input("Malic Acid", value=2.3)
        ash = st.number_input("Ash", value=2.3)
        alcalinity = st.number_input("Ash Alcalinity", value=19.0)
        
    with col2:
        magnesium = st.number_input("Magnesium", value=100.0)
        total_phenols = st.number_input("Total Phenols", value=2.2)
        flavanoids = st.number_input("Flavanoids", value=2.0)
        nonflavanoid = st.number_input("Nonflavanoid Phenols", value=0.3)
        
    with col3:
        proanthocyanins = st.number_input("Proanthocyanins", value=1.5)
        color_intensity = st.number_input("Color Intensity", value=5.0)
        hue = st.number_input("Hue", value=1.0)
        od280 = st.number_input("OD280/OD315", value=2.6)
        proline = st.number_input("Proline", value=700.0)

# --- PREDICTION ---
st.divider()
if st.button("🚀 Analyze Wine Cluster"):
    # Prepare data for prediction
    input_data = np.array([[
        alcohol, malic_acid, ash, alcalinity, magnesium, 
        total_phenols, flavanoids, nonflavanoid, proanthocyanins, 
        color_intensity, hue, od280, proline
    ]])
    
    prediction = model.predict(input_data)[0]
    
    # Show Result
    st.markdown(f"""
        <div class="cluster-box">
            This wine belongs to Cluster: {prediction}
        </div>
        """, unsafe_allow_html=True)
    
    # --- VISUALIZATION (Plotting same logic as original code) ---
    st.subheader("📊 Model Insight: Magnesium vs Proline")
    # Generating dummy data for visual context around the user's point
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # We plot a subtle marker for the user input
    ax.scatter(magnesium, proline, color='gold', s=300, marker='*', label="Your Wine", edgecolors='black', zorder=5)
    
    # Decorative cluster center visualization
    ax.scatter(model.cluster_centers_[:, 4], model.cluster_centers_[:, 12], 
               c=['red','blue','green','purple'], marker='x', s=100, label="Cluster Centers")
    
    ax.set_xlabel("Magnesium")
    ax.set_ylabel("Proline")
    ax.set_title(f"Prediction: Cluster {prediction}")
    ax.legend()
    st.pyplot(fig)

st.sidebar.markdown("### 🍷 About the Model")
st.sidebar.info("""
This application uses a K-Means Clustering model (K=4) to categorize wine based on its chemical composition. 
The dataset includes properties like Alcohol content, Phenols, and Color Intensity.
""")
