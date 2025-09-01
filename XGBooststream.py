import streamlit as st
import pandas as pd
import xgboost as xgb
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

# --- Load Data and Model ---
data = pd.read_excel("DATA.xlsx")
categorical_features = ['Biomass', 'Pollutant']
X = data.drop('Qe (mg/g)', axis=1)
y = data['Qe (mg/g)']

for col in categorical_features:
    X[col] = X[col].astype('category')

model = xgb.XGBRegressor(
    n_estimators=2994,
    max_depth=15,
    learning_rate=0.09891935310411579,
    subsample=0.5847597533737711,
    colsample_bytree=0.6636911331793705,
    tree_method='hist',
    enable_categorical=True,
    eval_metric='rmse'
)
model.fit(X, y)

# --- Metadata ---
feature_names = list(X.columns)
biomass_values = ['Swine manure', 'Dairy manure', 'Chicken manure']
pollutant_values = ['Cu(II)', 'Pb(II)', 'Cd(II)', 'U(VI)', 'Sb(III)', 'Zn(II)', 'Al(III)']

categories = {
    "🌱 Pyrolysis condition": (0, 4),
    "🧪 Biochar characteristics": (4, 14),
    "🧪 Adsorption experimental condition": (14, 21)
}

# Names for elemental inputs and ratios
C_COL = 'C (%)'
H_COL = 'H (%)'
O_COL = 'O (%)'
HC_COL = 'H/C'
OC_COL = 'O/C'

def compute_ratios_from_state():
    try:
        C = float(st.session_state.get(C_COL, 0.0))
        H = float(st.session_state.get(H_COL, 0.0))
        O = float(st.session_state.get(O_COL, 0.0))
        if C > 0:
            st.session_state[HC_COL] = round(H / C, 4)
            st.session_state[OC_COL] = round(O / C, 4)
        else:
            st.session_state[HC_COL] = 0.0
            st.session_state[OC_COL] = 0.0
    except Exception:
        pass

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
    # Ensure ratios are consistent if C/H/O changed
    compute_ratios_from_state()
    st.rerun()

# --- Form and Inputs ---
input_data = {}
with st.form("prediction_form"):
    # Recompute ratios on every run so disabled fields reflect current state
    compute_ratios_from_state()

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
                elif feature in (HC_COL, OC_COL):
                    # Auto-computed, read-only
                    val = float(st.session_state.get(feature, 0.0))
                    st.number_input(feature, value=float(val), format="%.4f", key=feature + "_disabled", disabled=True)
                    input_data[feature] = val
                else:
                    default = st.session_state.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=float(default), format="%.4f", key=feature)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    try:
        # Use submitted values directly (do NOT overwrite session_state for widget keys)
        C = float(input_data.get(C_COL, 0.0))
        H = float(input_data.get(H_COL, 0.0))
        O = float(input_data.get(O_COL, 0.0))

        # Compute ratios locally and mirror to session_state for display (safe: no widget uses these keys)
        HC = round(H / C, 4) if C > 0 else 0.0
        OC = round(O / C, 4) if C > 0 else 0.0
        input_data[HC_COL] = HC
        input_data[OC_COL] = OC
        st.session_state[HC_COL] = HC
        st.session_state[OC_COL] = OC

        # Validation
        vals_numeric = [v for v in input_data.values() if isinstance(v, (int, float))]
        if any(val < 0 for val in vals_numeric):
            st.error("❌ Inputs must be non-negative numbers.")
        elif C <= 0:
            st.error("❌ C (%) must be greater than 0 to compute H/C and O/C.")
        else:
            expected_HC = H / C
            expected_OC = O / C
            if abs(OC - expected_OC) > 0.10 * abs(expected_OC):
                st.error("❌ O/C ratio does not match the values.")
            elif abs(HC - expected_HC) > 0.10 * abs(expected_HC):
                st.error("❌ H/C ratio does not match the values.")
            else:
                df_input = pd.DataFrame([input_data])
                for col in categorical_features:
                    df_input[col] = pd.Categorical(df_input[col], categories=X[col].cat.categories)
                pred = model.predict(df_input)[0]
                st.success(f"🌟 Predicted Qe (mg/g): {pred:.2f}")
    except Exception as e:
        st.error(f"⚠️ Invalid input! {str(e)}")

