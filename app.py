import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    h1, h3 {
        color: #4b4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #4b4b4b;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    .prediction-result {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #4b4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.title("Prediksi Feedback Pelanggan Online Food")

st.markdown("""
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Input pengguna menggunakan kolom untuk tata letak yang lebih baik
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
    occupation = st.selectbox('Occupation', ['Student', 'Employee', '
