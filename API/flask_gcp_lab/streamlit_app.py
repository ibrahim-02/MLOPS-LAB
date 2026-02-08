import streamlit as st
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Wine Classifier",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS FOR STYLING
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #A52A2A;
        border-color: #A52A2A;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONFIGURATION
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Wine_grapes03.jpg/1200px-Wine_grapes03.jpg", caption="Wine Grapes")
    st.title("🍷 About the App")
    st.info(
        """
        This machine learning app predicts the **class** of wine based on its chemical composition.
        
        The model uses the Wine dataset to distinguish between:
        - **Class 0**
        - **Class 1**
        - **Class 2**
        
        Input 13 chemical features to get a prediction.
        """
    )
    st.write("---")
    st.caption("Built with Streamlit & Cloud Run")

# 4. MAIN APP INTERFACE
st.title("🍇 Wine Classification Predictor")
st.markdown("Adjust the sliders below to input the wine's chemical measurements.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Basic Properties", "Phenolic Compounds", "Additional Features"])

with tab1:
    st.subheader("Basic Chemical Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        alcohol = st.slider('Alcohol (%)', 11.0, 15.0, 13.2, 0.1)
        malic_acid = st.slider('Malic Acid (g/L)', 0.5, 6.0, 2.77, 0.1)
        ash = st.slider('Ash (g/L)', 1.0, 4.0, 2.51, 0.1)
    
    with col2:
        alcalinity_of_ash = st.slider('Alcalinity of Ash', 10.0, 30.0, 18.5, 0.5)
        magnesium = st.slider('Magnesium (mg/L)', 70.0, 162.0, 96.0, 1.0)
        proline = st.slider('Proline (mg/L)', 278.0, 1680.0, 820.0, 10.0)

with tab2:
    st.subheader("Phenolic Compounds")
    col3, col4 = st.columns(2)
    
    with col3:
        total_phenols = st.slider('Total Phenols', 0.5, 4.0, 2.20, 0.1)
        flavanoids = st.slider('Flavanoids', 0.0, 6.0, 2.53, 0.1)
        nonflavanoid_phenols = st.slider('Nonflavanoid Phenols', 0.1, 1.0, 0.26, 0.01)
    
    with col4:
        proanthocyanins = st.slider('Proanthocyanins', 0.4, 4.0, 1.56, 0.1)
        color_intensity = st.slider('Color Intensity', 1.0, 13.0, 5.0, 0.5)

with tab3:
    st.subheader("Color & Dilution Properties")
    col5, col6 = st.columns(2)
    
    with col5:
        hue = st.slider('Hue', 0.5, 1.8, 1.05, 0.01)
    
    with col6:
        od280_od315_of_diluted_wines = st.slider('OD280/OD315 of Diluted Wines', 1.0, 4.0, 3.33, 0.1)

st.write("---")

# 5. PREDICTION LOGIC
if st.button('🔍 Predict Wine Class'):
    
    # Visual loading spinner
    with st.spinner('Analyzing wine composition...'):
        data = {
            'alcohol': alcohol,
            'malic_acid': malic_acid,
            'ash': ash,
            'alcalinity_of_ash': alcalinity_of_ash,
            'magnesium': magnesium,
            'total_phenols': total_phenols,
            'flavanoids': flavanoids,
            'nonflavanoid_phenols': nonflavanoid_phenols,
            'proanthocyanins': proanthocyanins,
            'color_intensity': color_intensity,
            'hue': hue,
            'od280_od315_of_diluted_wines': od280_od315_of_diluted_wines,
            'proline': proline
        }
        
        try:
            # API Call - UPDATE THIS URL WITH YOUR DEPLOYED SERVICE URL
            response = requests.post('https://wine-classifier-160071191552.us-east1.run.app/predict', json=data)
            
            if response.status_code == 200:
                prediction = response.json()['prediction']
                
                # Dynamic Result Display
                st.success("Prediction Complete!")
                
                # Layout for result
                res_col1, res_col2 = st.columns([1, 2])
                
                # Image mapping for results
                images = {
                    "class_0": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Red_Wine_Glas.jpg/800px-Red_Wine_Glas.jpg",
                    "class_1": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/White_Wine_Glas.jpg/800px-White_Wine_Glas.jpg",
                    "class_2": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Wine_grapes03.jpg/1200px-Wine_grapes03.jpg"
                }
                
                # Class descriptions
                descriptions = {
                    "class_0": "High-quality wine with rich phenolic content",
                    "class_1": "Medium-bodied wine with balanced characteristics",
                    "class_2": "Light wine with distinctive flavor profile"
                }
                
                img_url = images.get(prediction, images["class_0"])
                description = descriptions.get(prediction, "Wine classification")

                with res_col1:
                    st.image(img_url, use_column_width=True)
                
                with res_col2:
                    st.header(f"Predicted: **{prediction.upper()}**!")
                    st.markdown(f"*{description}*")
                    st.markdown(f"""
                    **Key Features:**
                    * **Alcohol:** {alcohol}%
                    * **Flavanoids:** {flavanoids}
                    * **Color Intensity:** {color_intensity}
                    * **Proline:** {proline} mg/L
                    """)
                    st.balloons()
            
            else:
                st.error(f'Server Error: {response.status_code}')
                
        except requests.exceptions.RequestException as e:
            st.error('Connection Error: Could not reach the prediction service.')
            st.info('Make sure your Flask API is running on http://127.0.0.1:8080')