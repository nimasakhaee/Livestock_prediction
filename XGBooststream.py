import streamlit as st
import pandas as pd
import xgboost as xgb  # keeps predict path happy if model references xgb types
import joblib
import random

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

# --- Cached loaders ---
@st.cache_data
def load_data():
    return pd.read_excel("DATA.xlsx")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

data = load_data()
model = load_model()

# --- Setup data frame and metadata (no training here) ---
categorical_features = ['Biomass', 'Pollutant']
X = data.drop('Qe (mg/g)', axis=1)
y = data['Qe (mg/g)']

for col in categorical_features:
    X[col] = X[col].astype('category')

# Use categories from the data (reflects whatever is in your Excel, e.g., "Cattle manure")
biomass_values = X['Biomass'].cat.categories.tolist()
pollutant_values = X['Pollutant'].cat.categories.tolist()
feature_names = list(X.columns)

# Keep your original groups
categories = {
    "🌱 Pyrolysis condition": (0, 4),
    "🧪 Biochar characteristics": (4, 14),
    "🧪 Adsorption experimental condition": (14, 21)
}

# Element & ratio columns
C_COL = 'C (%)'
H_COL = 'H (%)'
O_COL = 'O (%)'
HC_COL = 'H/C'
OC_COL = 'O/C'

def compute_ratios_from_current_values(C, H, O):
    if C and C > 0:
        return round(H / C, 4), round(O / C, 4)
    return 0.0, 0.0

def within_pct(measured, expected, pct=0.10):
    if expected == 0:
        return measured == 0
    return abs(measured - expected) <= pct * abs(expected)

# --- Title ---
st.title("🌿 Qe (mg/g) Predictor - XGBoost")

# --- Random Value Button ---
if st.button("🎲 Fill with Random Values"):
    random_row = data.sample(1).iloc[0]
    for feature in feature_names:
        st.session_state[feature] = random_row[feature]
    # Keep derived ratios coherent
    C = float(random_row.get(C_COL, 0.0))
    H = float(random_row.get(H_COL, 0.0))
    O = float(random_row.get(O_COL, 0.0))
    HC, OC = compute_ratios_from_current_values(C, H, O)
    st.session_state[HC_COL] = HC
    st.session_state[OC_COL] = OC
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
                    default = st.session_state.get(feature, biomass_values[0] if biomass_values else "")
                    input_data[feature] = st.selectbox(
                        feature,
                        biomass_values,
                        index=biomass_values.index(default) if default in biomass_values else 0,
                        key=feature
                    )
                elif feature == 'Pollutant':
                    default = st.session_state.get(feature, pollutant_values[0] if pollutant_values else "")
                    input_data[feature] = st.selectbox(
                        feature,
                        pollutant_values,
                        index=pollutant_values.index(default) if default in pollutant_values else 0,
                        key=feature
                    )
                elif feature in (HC_COL, OC_COL):
                    # Read-only computed fields: show but disable
                    # Compute from current widget state if available
                    C = float(st.session_state.get(C_COL, 0.0))
                    H = float(st.session_state.get(H_COL, 0.0))
                    O = float(st.session_state.get(O_COL, 0.0))
                    HC, OC = compute_ratios_from_current_values(C, H, O)
                    # Mirror to session_state so display stays consistent
                    st.session_state[HC_COL] = HC
                    st.session_state[OC_COL] = OC
                    val = HC if feature == HC_COL else OC
                    st.number_input(feature, value=float(val), format="%.4f", key=feature + "_disabled", disabled=True)
                    input_data[feature] = val
                else:
                    default = st.session_state.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=float(default), format="%.4f", key=feature)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    try:
        # Pull elemental values from submitted inputs
        C = float(input_data.get(C_COL, 0.0))
        H = float(input_data.get(H_COL, 0.0))
        O = float(input_data.get(O_COL, 0.0))

        # Compute ratios locally (do NOT overwrite widget keys in session_state)
        HC = round(H / C, 4) if C > 0 else 0.0
        OC = round(O / C, 4) if C > 0 else 0.0
        input_data[HC_COL] = HC
        input_data[OC_COL] = OC

        # Keep the disabled displays synced (safe: these keys are not widget keys)
        st.session_state[HC_COL] = HC
        st.session_state[OC_COL] = OC

        # Basic validation
        vals_numeric = [v for v in input_data.values() if isinstance(v, (int, float))]
        if any(val < 0 for val in vals_numeric):
            st.error("❌ Inputs must be non-negative numbers.")
        elif C <= 0:
            st.error("❌ C (%) must be greater than 0 to compute H/C and O/C.")
        else:
            if not within_pct(OC, O / C, 0.10):
                st.error("❌ O/C ratio does not match the values.")
            elif not within_pct(HC, H / C, 0.10):
                st.error("❌ H/C ratio does not match the values.")
            else:
                df_input = pd.DataFrame([input_data])

                # Align categorical dtypes to training categories
                for col in categorical_features:
                    df_input[col] = pd.Categorical(df_input[col], categories=X[col].cat.categories)

                pred = model.predict(df_input)[0]
                st.success(f"🌟 Predicted Qe (mg/g): {pred:.2f}")
    except Exception as e:
        st.error(f"⚠️ Invalid input! {str(e)}")



