import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(layout="wide")

st.title("Dashboard Prediksi Hasil Panen Sawit")

#load data
df = pd.read_csv("dataset_kelapa_sawit_500.csv")

st.subheader("📊 Data Awal")
st.dataframe(df.head())

#pisah data
X = df.drop(["ID", "Hasil_Panen_ton_per_ha"], axis=1)
y = df["Hasil_Panen_ton_per_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#bangun model
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#matriks
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("R2 Score", round(r2, 3))
col2.metric("MAE", round(mae, 3))

#perbandingan grafik prediksi dan aktual
st.subheader(" Perbandingan Grafik Prediksi dan Aktual")

fig = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Aktual")
plt.ylabel("Prediksi")
plt.title("Perbandingan Prediksi dan Aktual")
st.pyplot(fig)

#grafik pengaruh fitur
st.subheader(" Pengaruh Fitur")

importance = model.feature_importances_
feature_names = X.columns

fig2 = plt.figure()
plt.barh(feature_names, importance)
plt.title("Feature Importance")
st.pyplot(fig2)

#user menginput
st.subheader("Prediksi Manual")

col1, col2, col3 = st.columns(3)

curah_hujan = col1.number_input("Curah Hujan", 0, 5000, 2000)
suhu = col2.number_input("Suhu", 0, 50, 28)
kelembaban = col3.number_input("Kelembaban", 0, 100, 80)

ndvi = col1.number_input("NDVI", 0.0, 1.0, 0.7)
umur = col2.number_input("Umur", 0, 30, 5)
luas = col3.number_input("Luas", 0, 100, 2)

pupuk = st.number_input("Pupuk", 0, 1000, 350)

#kategori nilai hasil panen
def kategori(nilai):
    if nilai < 7:
        return "Rendah"
    elif nilai <= 9:
        return "Sedang"
    else:
        return "Tinggi"

#prediksi
if st.button("Prediksi"):
    data = [[curah_hujan, suhu, kelembaban, ndvi, umur, luas, pupuk]]
    data = scaler.transform(data)

    hasil = model.predict(data)[0]
    kat = kategori(hasil)

    st.success(f"Hasil: {hasil:.2f} ton/ha")
    st.info(f"Kategori: {kat}")

#hasil analisis
st.subheader("Insight Analisis")

if r2 > 0.8:
    kualitas = "Sangat Baik"
elif r2 > 0.6:
    kualitas = "Cukup Baik"
else:
    kualitas = "Kurang Baik"

fitur_terpenting = feature_names[importance.argmax()]

st.write(f"""
### Kesimpulan Model:
- Kualitas Model: **{kualitas}**
- Model mampu menjelaskan sekitar **{round(r2*100,1)}%** data
- Rata-rata error: **{round(mae,2)} ton/ha**

### Insight Data:
- Faktor paling berpengaruh: **{fitur_terpenting}**
- Semakin optimal faktor ini → hasil panen meningkat

### Interpretasi:
- Model sudah cukup akurat untuk prediksi awal
- Bisa digunakan sebagai alat bantu keputusan
""")