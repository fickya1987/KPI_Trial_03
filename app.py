import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

    # Filter only necessary columns
    df = df[['TARGET TW TERKAIT', 'BOBOT', 'POLARITAS', 'NAMA KPI', 'POSISI PEKERJA', 'REALISASI TW TERKAIT']]
    df = df.dropna()

    # Use one-hot encoding for categorical columns
    df_encoded = pd.get_dummies(df, columns=['POLARITAS', 'NAMA KPI', 'POSISI PEKERJA'])

    # Define features and target
    X = df_encoded.drop('REALISASI TW TERKAIT', axis=1)
    y = df_encoded['REALISASI TW TERKAIT']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("Predict New KPI Realization")

    # Collect user inputs
    target_tw = st.number_input('Target TW Terkait', value=75)
    bobot = st.number_input('Bobot KPI', value=20)
    polaritas = st.selectbox('Polaritas', df['POLARITAS'].unique())
    nama_kpi = st.selectbox('Nama KPI', df['NAMA KPI'].unique())
    posisi = st.selectbox('Posisi Pekerja', df['POSISI PEKERJA'].unique())

    # Create input dataframe for prediction
    input_dict = {
        'TARGET TW TERKAIT': target_tw,
        'BOBOT': bobot
    }

    # Add encoded columns with 0 as default
    for col in X.columns:
        input_dict[col] = 0

    # Set selected category to 1
    input_dict['POLARITAS_' + polaritas] = 1
    input_dict['NAMA KPI_' + nama_kpi] = 1
    input_dict['POSISI PEKERJA_' + posisi] = 1

    # Align input to training columns
    input_df = pd.DataFrame([input_dict])[X.columns]

    # Predict and show result
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Realisasi TW Terkait: {prediction:.2f}")
