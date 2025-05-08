import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("KPI Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your KPI CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Preprocessing
    df = df.dropna(subset=['TARGET TW TERKAIT', 'REALISASI TW TERKAIT'])
    df['TARGET TW TERKAIT'] = pd.to_numeric(df['TARGET TW TERKAIT'], errors='coerce')
    df['REALISASI TW TERKAIT'] = pd.to_numeric(df['REALISASI TW TERKAIT'], errors='coerce')
    df = df.dropna()

    features = ['TARGET TW TERKAIT', 'BOBOT', 'POLARITAS', 'NAMA KPI', 'POSISI PEKERJA']
    df = df[features + ['REALISASI TW TERKAIT']]

    # Encode categorical features
    for col in ['POLARITAS', 'NAMA KPI', 'POSISI PEKERJA']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[features]
    y = df['REALISASI TW TERKAIT']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Predict New KPI Realization")
    input_data = {
        'TARGET TW TERKAIT': st.number_input('Target TW Terkait', value=75),
        'BOBOT': st.number_input('Bobot KPI', value=20),
        'POLARITAS': st.selectbox('Polaritas', df['POLARITAS'].unique()),
        'NAMA KPI': st.selectbox('Nama KPI', df['NAMA KPI'].unique()),
        'POSISI PEKERJA': st.selectbox('Posisi Pekerja', df['POSISI PEKERJA'].unique())
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Realisasi TW Terkait: {prediction:.2f}")
