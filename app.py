import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.title("📊 KPI Prediction & Analysis App")

# Upload file
uploaded_file = st.file_uploader("📂 Upload KPI Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    df = df.dropna(subset=['TARGET TW TERKAIT', 'REALISASI TW TERKAIT'])
    df['TARGET TW TERKAIT'] = pd.to_numeric(df['TARGET TW TERKAIT'], errors='coerce')
    df['REALISASI TW TERKAIT'] = pd.to_numeric(df['REALISASI TW TERKAIT'], errors='coerce')
    df = df.dropna()

    df = df[['TARGET TW TERKAIT', 'BOBOT', 'POLARITAS', 'NAMA KPI', 'POSISI PEKERJA', 'REALISASI TW TERKAIT']]

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['POLARITAS', 'NAMA KPI', 'POSISI PEKERJA'])

    # Features and target
    X = df_encoded.drop('REALISASI TW TERKAIT', axis=1)
    y = df_encoded['REALISASI TW TERKAIT']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.metric("R² Score", f"{r2:.2f}")
    st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Realisasi')
    st.pyplot(fig)

    st.subheader("🧠 Predict a New KPI Outcome")

    # User Input
    target_tw = st.number_input('🎯 Target TW Terkait', value=75)
    bobot = st.number_input('⚖️ Bobot KPI', value=20)
    polaritas = st.selectbox('📌 Polaritas', df['POLARITAS'].unique())
    nama_kpi = st.selectbox('📝 Nama KPI', df['NAMA KPI'].unique())
    posisi = st.selectbox('👤 Posisi Pekerja', df['POSISI PEKERJA'].unique())

    # Prediction Input
    input_dict = {col: 0 for col in X.columns}
    input_dict['TARGET TW TERKAIT'] = target_tw
    input_dict['BOBOT'] = bobot
    input_dict[f'POLARITAS_{polaritas}'] = 1
    input_dict[f'NAMA KPI_{nama_kpi}'] = 1
    input_dict[f'POSISI PEKERJA_{posisi}'] = 1

    input_df = pd.DataFrame([input_dict])[X.columns]
    predicted_value = model.predict(input_df)[0]

    st.success(f"✅ Predicted Realisasi TW Terkait: {predicted_value:.2f}")

