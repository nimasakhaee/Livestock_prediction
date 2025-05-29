import streamlit as st
import pandas as pd
import xgboost as xgb
import random
import joblib

# --- Page setup ---
st.set_page_config(page_title="Qe (mg/g) Predictor - XGBoost", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f2f5f7; }
        div[data-testid="stSidebar"] { background-color: #e0ecf1; }
        .stButton > button {
            background-color: #6fa8dc;
            color: white;
            font-weight: bold;
            padding: 10px 24px;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_excel("DATA.xlsx")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

# --- Load Data & Model ---
data = load_data()
model = load_model()

categorical_features = ['Biomass', 'Pollutant']
X = data.drop('Qe (mg/g)', axis=1)
y = data['Qe (mg/g)']
for col in categorical_features:
    X[col] = X[col].astype('category')
    

# --- Metadata ---
feature_names = list(X.columns)
biomass_values = ['Swine manure', 'Dairy manure', 'Chicken manure']
pollutant_values = ['Cu(II)', 'Pb(II)', 'Cd(II)', 'U(VI)', 'Sb(III)', 'Zn(II)', 'Al(III)']

categories = {
    "🌱 Pyrolysis condition": (0, 4),
    "🧪 Biochar characteristics": (4, 14),
    "🧪 Adsorption experimental condition": (14, 21)
}

# --- Title ---
st.title("🌿 Qe (mg/g) Predictor - XGBoost")

# --- Random Value Button ---
if st.button("🎲 Fill with Random Values"):
    random_row = data.sample(1).iloc[0]
    for feature in feature_names:
        st.session_state[feature] = random_row[feature]
    st.rerun()

# --- Form and Inputs ---
input_data = {}
with st.form("prediction_form"):
    cols = st.columns(len(categories))
    for idx, (cat_name, (start, end)) in enumerate(categories.items()):
        with cols[idx]:
            st.markdown(f"### {cat_name}")
            for feature in feature_names[start:end]:
                if feature == 'Biomass':
                    default = st.session_state.get(feature, biomass_values[0])
                    input_data[feature] = st.selectbox(feature, biomass_values, index=biomass_values.index(default) if default in biomass_values else 0, key=feature)
                elif feature == 'Pollutant':
                    default = st.session_state.get(feature, pollutant_values[0])
                    input_data[feature] = st.selectbox(feature, pollutant_values, index=pollutant_values.index(default) if default in pollutant_values else 0, key=feature)
                else:
                    default = st.session_state.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=float(default), format="%.4f", key=feature)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    try:
        vals = list(input_data.values())
        if (vals[7]/vals[5])*0.9 >= vals[10] or (vals[7]/vals[5])*1.1 <= vals[10]:
            st.error("❌ O/C ratio does not match the values")
        elif (vals[8]/vals[5])*0.9 >= vals[11] or (vals[8]/vals[5])*1.1 <= vals[11]:
            st.error("❌ H/C ratio does not match the values")
        elif any(val < 0 for val in vals if isinstance(val, (int, float))):
            st.error("❌ Inputs must be non-negative numbers.")
        else:
            df_input = pd.DataFrame([input_data])
            for col in categorical_features:
                df_input[col] = pd.Categorical(df_input[col], categories=X[col].cat.categories)
            pred = model.predict(df_input)[0]
            st.success(f"🌟 Predicted Qe (mg/g): {pred:.2f}")
    except Exception as e:
        st.error(f"⚠️ Invalid input! {str(e)}")
