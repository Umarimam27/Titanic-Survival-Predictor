# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import base64
# import time 

# # ================================
# # 🎯 CONFIGURATION (Updated for Titanic)
# # ================================
# # NOTE: Ensure you have a file named 'titanic_lr_model.pkl' in the same directory.
# MODEL_PATH = "lr_model.pkl" 
# FEATURE_NAMES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']

# # BASE64 FALLBACK IMAGE (Dark, subtle, embedded SVG image)
# BASE64_FALLBACK_IMAGE = """
# data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwMCIgaGVpZ2h0PSI4MDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+
#   <rect width="1200" height="800" fill="#1b1c19"/>
#   <rect x="50" y="50" width="1100" height="700" stroke="#0077b6" stroke-width="5" fill="none" stroke-opacity="0.3"/>
#   <rect x="100" y="100" width="500" height="200" fill="rgba(255, 255, 255, 0.05)" rx="10" ry="10"/>
#   <rect x="600" y="400" width="500" height="300" fill="rgba(255, 255, 255, 0.08)" rx="10" ry="10"/>
#   <line x1="50" y1="750" x2="1150" y2="50" stroke="#0077b6" stroke-width="2" stroke-opacity="0.1"/>
#   <circle cx="600" cy="400" r="100" fill="rgba(255, 255, 255, 0.05)"/>
# </svg>
# """

# # ================================
# # ⚙️ UTILITIES
# # ================================

# def get_base64_image_url(uploaded_file):
#     """Reads an uploaded file and converts it to a Base64 data URL."""
#     try:
#         bytes_data = uploaded_file.getvalue()
#         base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
#         mime_type = uploaded_file.type if uploaded_file.type else "image/png"
#         return f"data:{mime_type};base64,{base64_encoded_data}"
#     except Exception as e:
#         st.error(f"Error processing image: {e}")
#         return None

# # ================================
# # 🌄 CINEMATIC BACKGROUND SLIDESHOW
# # ================================

# def set_cinematic_bg(base64_urls, interval_per_image=6):
#     """Applies a smooth crossfading background slideshow using Base64 URLs."""
#     num_images = len(base64_urls)
#     total_duration = num_images * interval_per_image

#     # Overlay opacity set to 0.6 for better readability against dark theme
#     OVERLAY_OPACITY = "rgba(0,0,0,0.6)" 

#     # --- CSS Selectors for Frosted Glass Effect ---
#     FROSTED_GLASS_SELECTORS = """
#         /* Sidebar container */
#         [data-testid="stSidebar"] > div:first-child,
#         /* Tab containers */
#         [data-testid="stTabs"] > div:nth-child(2)
#     """

#     if num_images == 0:
#         st.warning("No images uploaded. Using static dark background image.")
        
#         # --- FALLBACK CSS: Base64 Embedded Image (Static) ---
#         st.markdown(f"""
#             <style>
#             .stApp {{
#                 background-image: url('{BASE64_FALLBACK_IMAGE.replace('\n', '')}');
#                 background-attachment: fixed;
#                 background-size: cover;
#                 background-position: center;
#                 animation: none !important;
#             }}
#             .stApp::before {{
#                 content: "";
#                 position: fixed;
#                 top: 0; left: 0;
#                 width: 100%; height: 100%;
#                 background: {OVERLAY_OPACITY};
#                 z-index: 0;
#             }}
#             {FROSTED_GLASS_SELECTORS} {{
#                 background: rgba(255, 255, 255, 0.15); 
#                 backdrop-filter: blur(8px);
#                 border-radius: 16px;
#                 padding: 20px;
#                 z-index: 10;
#             }}
#             </style>
#         """, unsafe_allow_html=True)
#         return

#     # --- SUCCESS CSS: Image Slideshow (Dynamic) ---
    
#     css_keyframes = []
#     for i in range(num_images):
#         start_percent = (i * 100) / num_images
#         hold_percent = (start_percent + ((100 / num_images) * (1 - 1 / interval_per_image))) 
        
#         css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
#         if i < num_images - 1:
#              css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
        
#     css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")


#     st.markdown(f"""
#         <style>
#         .stApp {{
#             background-size: cover;
#             background-attachment: fixed;
#             background-repeat: no-repeat;
#             background-image: url('{base64_urls[0]}');
#             animation: cinematicBg {total_duration}s infinite;
#         }}
#         @keyframes cinematicBg {{
#             {"".join(css_keyframes)}
#         }}

#         /* Apply a dark overlay for better readability and subtlety */
#         .stApp::before {{
#             content: "";
#             position: fixed;
#             top: 0; left: 0;
#             width: 100%; height: 100%;
#             background: {OVERLAY_OPACITY};
#             z-index: 0;
#         }}
        
#         /* Frosted glass content containers (Only Sidebar and Tabs) */
#         {FROSTED_GLASS_SELECTORS} {{
#             background: rgba(255, 255, 255, 0.15); 
#             backdrop-filter: blur(8px);
#             border-radius: 16px;
#             padding: 20px;
#             z-index: 10;
#         }}

#         /* General styling adjustments */
#         * {{ color: white; font-family: 'Inter', sans-serif; }}
#         /* Ensure prediction box stands out */
#         .prediction-box {{
#             color: white !important; 
#         }}
#         .prediction-box h2, .prediction-box h1, .prediction-box p {{
#             color: inherit !important;
#         }}

#         [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
#         </style>
#     """, unsafe_allow_html=True)

# # ================================
# # 📌 LOAD MODEL
# # ================================
# model = None
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
# except FileNotFoundError:
#     st.error(f"❌ Model file '{MODEL_PATH}' not found. Please ensure you have trained and saved the model as `titanic_lr_model.pkl`.")
# except Exception as e:
#     st.error(f"⚠️ Error loading model: {e}")

# # ================================
# # 📌 SIDEBAR CONTENT (Includes Image Uploader)
# # ================================

# base64_image_urls = []
# with st.sidebar:
#     st.title("ℹ️ App Configuration")
    
#     # --- IMAGE UPLOADER ---
#     st.subheader("🖼️ Background Images")
#     uploaded_files = st.file_uploader(
#         "Upload images (JPG/PNG) for the slideshow:",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#         help="The slideshow starts once images are uploaded. This runs without a network."
#     )
    
#     if uploaded_files:
#         if len(uploaded_files) < 3:
#             st.info("Upload at least 3 images for a better slideshow effect.")
        
#         with st.spinner(f"Processing {len(uploaded_files)} image(s)..."):
#             for file in uploaded_files:
#                 url = get_base64_image_url(file)
#                 if url:
#                     base64_image_urls.append(url)
#             time.sleep(0.5)

#     st.markdown("---")
#     st.subheader("Model Info")
#     st.info("This app predicts a Survival Score for Titanic passengers using a trained Linear Regression Model.")
#     st.markdown(
#         "🚢 Model adapted from your **Titanic Data Analysis** notebook.",
#         unsafe_allow_html=True
#     )

# # Apply background logic (using images uploaded in the sidebar)
# set_cinematic_bg(base64_image_urls)

# # ================================
# # 🚢 HEADER (Main Screen)
# # ================================
# st.markdown("<h1 style='text-align:center; color:#ADD8E6; text-shadow: 2px 2px 6px #000000;'>⚓ TITANIC SURVIVAL SCORE PREDICTOR</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; font-size:18px; color:#F0F0F0;'>Estimate a passenger's chance of survival based on key features.</p>", unsafe_allow_html=True)

# # ================================
# # 📊 TABS (Main Screen)
# # ================================
# tab1, tab2 = st.tabs(["🔑 Prediction", "📈 Model Info"])

# # ================================
# # ✨ TAB 1 — PREDICTION
# # ================================
# with tab1:
#     st.header("Enter Passenger Details")

#     if model:
#         col1, col2 = st.columns(2)

#         # Collect user inputs
#         with col1:
#             pclass = st.selectbox("🎫 Ticket Class (Pclass)", [1, 2, 3], index=2, help="1st=Upper, 2nd=Middle, 3rd=Lower")
#             sex_input = st.selectbox("🚻 Sex", ["Male", "Female"])
#             age = st.number_input("🎂 Age", 0, 100, 30)
#             fare = st.number_input("💲 Ticket Fare", 0.0, 500.0, 32.0, step=1.0)

#         with col2:
#             sibsp = st.number_input("👨‍👩‍👧‍👦 Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
#             parch = st.number_input("👶 Parents/Children Aboard (Parch)", 0, 9, 0)
#             embarked_input = st.selectbox("📍 Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"], index=0)

#         if st.button("🌊 Estimate Survival Score", use_container_width=True, type="primary"):
#             # --- Preprocessing Inputs to Model Features ---
            
#             # 1. Sex (1 for Male, 0 for Female - assuming male was the dummy variable)
#             sex_encoded = 1 if sex_input == "Male" else 0
            
#             # 2. Embarked (C is baseline: Q=0, S=0)
#             embarked_q = 1 if embarked_input == "Queenstown (Q)" else 0
#             embarked_s = 1 if embarked_input == "Southampton (S)" else 0
            
#             # Assemble the input data in the correct feature order
#             input_data = [pclass, sex_encoded, age, sibsp, parch, fare, embarked_q, embarked_s]
#             features_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
            
#             try:
#                 prediction = model.predict(features_df)[0]
                
#                 # Clamp prediction between 0 and 1 for interpretation
#                 survival_score = max(0, min(1, prediction)) 
#                 survival_percentage = survival_score * 100
                
#                 # Determine color and message based on score
#                 if survival_score >= 0.6:
#                     color_code = "#4CAF50" # Green
#                     qualitative_result = "High Estimated Chance of Survival"
#                     icon = "✅"
#                 elif survival_score >= 0.4:
#                     color_code = "#FFC300" # Yellow/Orange
#                     qualitative_result = "Moderate Estimated Chance of Survival"
#                     icon = "⚠️"
#                 else:
#                     color_code = "#D32F2F" # Red
#                     qualitative_result = "Low Estimated Chance of Survival"
#                     icon = "❌"

#                 st.markdown(f"""
#                     <div class="prediction-box" style="background-color:{color_code}; padding:20px; border-radius:15px; text-align:center; margin-top: 20px;">
#                         <h2 style="color:white;">{icon} Estimated Survival Score</h2>
#                         <h1 style="color:white;">{survival_percentage:.2f}%</h1>
#                         <p style="color:white; font-size: 1.2em;">{qualitative_result}</p>
#                     </div>
#                 """, unsafe_allow_html=True)
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")
#     else:
#         st.warning("⚠️ Prediction feature unavailable due to model loading error. Please check the model file path.")

# # ================================
# # 📊 TAB 2 — MODEL INFO
# # ================================
# with tab2:
#     st.header("Model Overview & Performance")

#     st.subheader("📌 Model Used")
#     st.info("**Linear Regression** was used. Note that Linear Regression on a binary target (Survival) is unusual; Logistic Regression is generally preferred for this type of problem.")

#     st.subheader("🧰 Training Features")
#     st.markdown(f"""
#     The model was trained on the following **8 features**:
#     - `Pclass`: Ticket class (1, 2, 3)
#     - `Sex`: Gender (1=Male, 0=Female)
#     - `Age`: Passenger age
#     - `SibSp`: # of siblings/spouses aboard
#     - `Parch`: # of parents/children aboard
#     - `Fare`: Ticket fare
#     - `Embarked_Q`, `Embarked_S`: One-hot encoded ports of embarkation (C is baseline)
#     """)

#     st.subheader("📊 Performance Metrics (Based on Notebook Snippet)")
#     # Using example metrics from the provided notebook for context
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Training R²", "0.4421", delta="Moderate Fit")
#     col2.metric("Testing R²", "0.4508", delta="Similar Performance")
#     col3.metric("Features Used", f"{len(FEATURE_NAMES)}", delta="8 total features")
    
#     st.markdown("---")
#     st.caption("Disclaimer: This model is for illustrative purposes only and does not reflect historical accuracy.")


## ================================ revised code ==================================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import time 

# ================================
# 🎯 CONFIGURATION (Updated for Titanic)
# ================================
# NOTE: Ensure you have a file named 'titanic_lr_model.pkl' in the same directory.
MODEL_PATH = "lr_model.pkl" 

# *** CRITICAL FIX: DEFINITIVE FEATURE ORDER CONFIRMED BY NOTEBOOK ANALYSIS ***
# This is the EXACT order of the features (X.columns) used to train the pickled model.
FEATURE_NAMES = [
    'Pclass', 'SibSp', 'Parch', 
    'log_Age', 'log_Fare', 
    'Sex_male', # Swapped position with Embarked dummies
    'Embarked_Q', 'Embarked_S' 
]

# BASE64 FALLBACK IMAGE (Dark, subtle, embedded SVG image)
BASE64_FALLBACK_IMAGE = """
data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwMCIgaGVpZ2h0PSI4MDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+
  <rect width="1200" height="800" fill="#1b1c19"/>
  <rect x="50" y="50" width="1100" height="700" stroke="#0077b6" stroke-width="5" fill="none" stroke-opacity="0.3"/>
  <rect x="100" y="100" width="500" height="200" fill="rgba(255, 255, 255, 0.05)" rx="10" ry="10"/>
  <rect x="600" y="400" width="500" height="300" fill="rgba(255, 255, 255, 0.08)" rx="10" ry="10"/>
  <line x1="50" y1="750" x2="1150" y2="50" stroke="#0077b6" stroke-width="2" stroke-opacity="0.1"/>
  <circle cx="600" cy="400" r="100" fill="rgba(255, 255, 255, 0.05)"/>
</svg>
"""

# ================================
# ⚙️ UTILITIES
# ================================

def get_base64_image_url(uploaded_file):
    """Reads an uploaded file and converts it to a Base64 data URL."""
    try:
        bytes_data = uploaded_file.getvalue()
        base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
        mime_type = uploaded_file.type if uploaded_file.type else "image/png"
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        return None

# ================================
# 🌄 CINEMATIC BACKGROUND SLIDESHOW
# ================================

def set_cinematic_bg(base64_urls, interval_per_image=6):
    """Applies a smooth crossfading background slideshow using Base64 URLs."""
    num_images = len(base64_urls)
    total_duration = num_images * interval_per_image

    # Overlay opacity set to 0.6 for better readability against dark theme
    OVERLAY_OPACITY = "rgba(0,0,0,0.6)" 

    # --- CSS Selectors for Frosted Glass Effect ---
    FROSTED_GLASS_SELECTORS = """
        /* Sidebar container */
        [data-testid="stSidebar"] > div:first-child,
        /* Tab containers */
        [data-testid="stTabs"] > div:nth-child(2)
    """

    if num_images == 0:
        st.warning("No images uploaded. Using static dark background image.")
        
        # --- FALLBACK CSS: Base64 Embedded Image (Static) ---
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url('{BASE64_FALLBACK_IMAGE.replace('\n', '')}');
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
                animation: none !important;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background: {OVERLAY_OPACITY};
                z-index: 0;
            }}
            {FROSTED_GLASS_SELECTORS} {{
                background: rgba(255, 255, 255, 0.15); 
                backdrop-filter: blur(8px);
                border-radius: 16px;
                padding: 20px;
                z-index: 10;
            }}
            </style>
        """, unsafe_allow_html=True)
        return

    # --- SUCCESS CSS: Image Slideshow (Dynamic) ---
    
    css_keyframes = []
    for i in range(num_images):
        start_percent = (i * 100) / num_images
        hold_percent = (start_percent + ((100 / num_images) * (1 - 1 / interval_per_image))) 
        
        css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
        if i < num_images - 1:
             css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
        
    css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")


    st.markdown(f"""
        <style>
        .stApp {{
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-image: url('{base64_urls[0]}');
            animation: cinematicBg {total_duration}s infinite;
        }}
        @keyframes cinematicBg {{
            {"".join(css_keyframes)}
        }}

        /* Apply a dark overlay for better readability and subtlety */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: {OVERLAY_OPACITY};
            z-index: 0;
        }}
        
        /* Frosted glass content containers (Only Sidebar and Tabs) */
        {FROSTED_GLASS_SELECTORS} {{
            background: rgba(255, 255, 255, 0.15); 
            backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 20px;
            z-index: 10;
        }}

        /* General styling adjustments */
        * {{ color: white; font-family: 'Inter', sans-serif; }}
        /* Ensure prediction box stands out */
        .prediction-box {{
            color: white !important; 
        }}
        .prediction-box h2, .prediction-box h1, .prediction-box p {{
            color: inherit !important;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
        </style>
    """, unsafe_allow_html=True)

# ================================
# 📌 LOAD MODEL
# ================================
model = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Please ensure you have trained and saved the model as `titanic_lr_model.pkl`.")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# ================================
# 📌 SIDEBAR CONTENT (Includes Image Uploader)
# ================================

base64_image_urls = []
with st.sidebar:
    st.title("ℹ️ App Configuration")
    
    # --- IMAGE UPLOADER ---
    st.subheader("🖼️ Background Images")
    uploaded_files = st.file_uploader(
        "Upload images (JPG/PNG) for the slideshow:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="The slideshow starts once images are uploaded. This runs without a network."
    )
    
    if uploaded_files:
        if len(uploaded_files) < 3:
            st.info("Upload at least 3 images for a better slideshow effect.")
        
        with st.spinner(f"Processing {len(uploaded_files)} image(s)..."):
            for file in uploaded_files:
                url = get_base64_image_url(file)
                if url:
                    base64_image_urls.append(url)
            time.sleep(0.5)

    st.markdown("---")
    st.subheader("Model Info")
    st.info("This app predicts a Survival Score for Titanic passengers using a trained Linear Regression Model.")
    st.markdown(
        "🚢✨ Made with ❤️<br>👨‍💻 Developed by **Umar Imam**",
        unsafe_allow_html=True
    )

# Apply background logic (using images uploaded in the sidebar)
set_cinematic_bg(base64_image_urls)

# ================================
# 🚢 HEADER (Main Screen)
# ================================
st.markdown("<h1 style='text-align:center; color:#ADD8E6; text-shadow: 2px 2px 6px #000000;'>⚓ TITANIC SURVIVAL SCORE PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#F0F0F0;'>Estimate a passenger's chance of survival based on key features.</p>", unsafe_allow_html=True)

# ================================
# 📊 TABS (Main Screen)
# ================================
tab1, tab2 = st.tabs(["🔑 Prediction", "📈 Model Info"])

# ================================
# ✨ TAB 1 — PREDICTION
# ================================
with tab1:
    st.header("Enter Passenger Details")

    if model:
        col1, col2 = st.columns(2)

        # Collect user inputs
        with col1:
            pclass = st.selectbox("🎫 Ticket Class (Pclass)", [1, 2, 3], index=2, help="1st=Upper, 2nd=Middle, 3rd=Lower")
            sex_input = st.selectbox("🚻 Sex", ["Male", "Female"])
            # Note on Age: Using 0.01 for log transformation to avoid math domain errors.
            age = st.number_input("🎂 Age", 0.01, 100.0, 30.0, format="%.2f") 
            # Note on Fare: Using 0.01 for log transformation to avoid math domain errors.
            fare = st.number_input("💲 Ticket Fare", 0.01, 500.0, 32.0, format="%.2f") 

        with col2:
            sibsp = st.number_input("👨‍👩‍👧‍👦 Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
            parch = st.number_input("👶 Parents/Children Aboard (Parch)", 0, 9, 0)
            embarked_input = st.selectbox("📍 Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"], index=0)

        if st.button("🌊 Estimate Survival Score", use_container_width=True, type="primary"):
            
            # --- START: CRITICAL PREPROCESSING TO MATCH MODEL TRAINING ---
            
            # 1. Feature Transformation (Logarithms)
            log_age = np.log(age)
            log_fare = np.log(fare)
            
            # 2. One-Hot Encoding for Categorical Features
            # Sex: The model expects 'Sex_male' (1 for Male, 0 for Female)
            sex_male = 1 if sex_input == "Male" else 0
            
            # Embarked: The model expects 'Embarked_Q' and 'Embarked_S' (Cherbourg (C) is the baseline/dropped dummy)
            embarked_q = 1 if embarked_input == "Queenstown (Q)" else 0
            embarked_s = 1 if embarked_input == "Southampton (S)" else 0
            
            # 3. Assemble the FINAL input data in the EXACT required order (Confirmed by Notebook)
            # Order: Pclass, SibSp, Parch, log_Age, log_Fare, Sex_male, Embarked_Q, Embarked_S
            input_data = [
                pclass, 
                sibsp, 
                parch, 
                log_age, 
                log_fare, 
                sex_male,     # Position corrected
                embarked_q, 
                embarked_s
            ]
            
            # Create DataFrame with the correct column names AND order for the model
            features_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
            
            # --- END: CRITICAL PREPROCESSING ---
            
            try:
                prediction = model.predict(features_df)[0]
                
                # Clamp prediction between 0 and 1 for interpretation
                survival_score = max(0, min(1, prediction)) 
                survival_percentage = survival_score * 100
                
                # Determine color and message based on score
                if survival_score >= 0.6:
                    color_code = "#4CAF50" # Green
                    qualitative_result = "High Estimated Chance of Survival"
                    icon = "✅"
                elif survival_score >= 0.4:
                    color_code = "#FFC300" # Yellow/Orange
                    qualitative_result = "Moderate Estimated Chance of Survival"
                    icon = "⚠️"
                else:
                    color_code = "#D32F2F" # Red
                    qualitative_result = "Low Estimated Chance of Survival"
                    icon = "❌"

                st.markdown(f"""
                    <div class="prediction-box" style="background-color:{color_code}; padding:20px; border-radius:15px; text-align:center; margin-top: 20px;">
                        <h2 style="color:white;">{icon} Estimated Survival Score</h2>
                        <h1 style="color:white;">{survival_percentage:.2f}%</h1>
                        <p style="color:white; font-size: 1.2em;">{qualitative_result}</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("⚠️ Prediction feature unavailable due to model loading error. Please check the model file path.")

# ================================
# 📊 TAB 2 — MODEL INFO
# ================================
with tab2:
    st.header("Model Overview & Performance")

    st.subheader("📌 Model Used")
    st.info("**Linear Regression** was used. Note that Linear Regression on a binary target (Survival) is unusual; **Logistic Regression** is generally preferred for this type of problem.")

    st.subheader("🧰 Training Features (The final features the model *expects* in order)")
    st.markdown(f"""
    The model was trained on the following **8 transformed features in this precise order**:
    1. `Pclass`
    2. `SibSp`
    3. `Parch`
    4. `log_Age` (Logarithm of Age)
    5. `log_Fare` (Logarithm of Fare)
    6. **`Sex_male`** (One-hot encoded, 1 if Male)
    7. `Embarked_Q` (One-hot encoded, 1 if Queenstown)
    8. `Embarked_S` (One-hot encoded, 1 if Southampton)
    """)

    # --- Defining results_df here ---
    results_data = {
        'Metric': ['Training R²', 'Testing R²', 'Features Used'],
        'Value': [0.4421, 0.4508, len(FEATURE_NAMES)]
    }
    results_df = pd.DataFrame(results_data)
    # --- END CRITICAL FIX ---

    # Display the model evaluation results
    st.subheader("📊 Model Evaluation Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.caption("Disclaimer: This model is for illustrative purposes only and does not reflect historical accuracy.")
