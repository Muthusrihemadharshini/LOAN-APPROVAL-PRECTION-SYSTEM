import streamlit as st
import numpy as np
import pickle


import requests
import pickle
import io


st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

def load_pickle_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

@st.cache_resource
def load_all():
    model = load_pickle_from_drive("1Chqgz6LMnk25_21kGr_xYbU7HdvA9aBo")
    scaler = load_pickle_from_drive("1KocsYONZ_2EHi6HiT7mfSrweDI0Tu7PJ")
    columns = load_pickle_from_drive("1sJfyexY6DVNS7WHnglcMwm7Fyjog4AS_")
    return model, scaler, columns

model, scaler, columns = load_all()

st.markdown("""
<style>

/* 🔘 BUTTON TEXT → FORCE WHITE */
.stButton > button {
    color: white !important;
}

/* 🧠 DROPDOWN MENU (options list) */
[data-baseweb="menu"] {
    background-color: #1e1e1e !important;  /* dark bg */
}

/* Dropdown options text */
[data-baseweb="menu"] * {
    color: white !important;
}

/* Selected + hover option */
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #333 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🏦 Loan Default Prediction System</p>', unsafe_allow_html=True)
st.write("Enter applicant details to assess loan risk")


income = st.number_input("Total Income", min_value=1)
credit = st.number_input("Loan Amount", min_value=1)
annuity = st.number_input("Annuity Amount", min_value=1)

age = st.slider("Age", 18, 70)
employment = st.slider("Employment Years", 0, 40)
family_members = st.slider("Family Members", 1, 10)

car = st.selectbox("Own Car", ["Yes", "No"])
house = st.selectbox("Own House", ["Yes", "No"])

income_type = st.selectbox("Income Type", ["Working", "Pensioner", "State servant"])
education = st.selectbox("Education", ["Secondary / secondary special", "Higher education"])
family_status = st.selectbox("Family Status", ["Married", "Single / not married"])
housing = st.selectbox("Housing Type", ["House / apartment", "Rented apartment"])


income_credit_ratio = income / credit
emi_ratio = annuity / income


input_dict = {col: 0 for col in columns}

input_dict["AMT_INCOME_TOTAL"] = income
input_dict["AMT_CREDIT"] = credit
input_dict["AMT_ANNUITY"] = annuity
input_dict["CNT_FAM_MEMBERS"] = family_members

input_dict["Age"] = age
input_dict["Employment_Years"] = employment

input_dict["DAYS_BIRTH"] = -age * 365
input_dict["DAYS_EMPLOYED"] = -employment * 365

input_dict["Income_Credit_Ratio"] = income_credit_ratio
input_dict["EMI_Ratio"] = emi_ratio

input_dict["AMT_GOODS_PRICE"] = credit * 0.9


if f"NAME_INCOME_TYPE_{income_type}" in columns:
    input_dict[f"NAME_INCOME_TYPE_{income_type}"] = 1

if f"NAME_EDUCATION_TYPE_{education}" in columns:
    input_dict[f"NAME_EDUCATION_TYPE_{education}"] = 1

if f"NAME_FAMILY_STATUS_{family_status}" in columns:
    input_dict[f"NAME_FAMILY_STATUS_{family_status}"] = 1

if f"NAME_HOUSING_TYPE_{housing}" in columns:
    input_dict[f"NAME_HOUSING_TYPE_{housing}"] = 1


input_dict["FLAG_OWN_CAR_Y"] = 1 if car == "Yes" else 0
input_dict["FLAG_OWN_REALTY_Y"] = 1 if house == "Yes" else 0


input_data = np.array([input_dict[col] for col in columns]).reshape(1, -1)
input_data = scaler.transform(input_data)


if st.button("🔍 Predict Risk"):

    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.2:
        st.error("🔴 High Risk")
    elif prob > 0.1:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    st.subheader(f"📊 Default Probability: {round(prob*100,2)}%")
    st.progress(int(prob * 100))