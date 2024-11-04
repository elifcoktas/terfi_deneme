import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Veriyi yükleme
file_path = 'employeePromotion.csv'  # CSV dosyasının doğru yolunu belirtin
data = pd.read_csv(file_path)

# Kategorik değişkenleri dönüştürme ve eksik değerleri doldurma
label_encoders = {}
for column in ['department', 'region', 'education', 'gender', 'recruitment_channel']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

data['previous_year_rating'].fillna(data['previous_year_rating'].median(), inplace=True)

# Özellikleri ve hedef değişkeni ayırma
X = data.drop(['employee_id', 'is_promoted'], axis=1)
y = data['is_promoted']

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit uygulaması
st.title("Çalışan Terfi Tahmin Uygulaması")

# Kullanıcı girdileri
department = st.selectbox("Departman", label_encoders['department'].classes_)
region = st.selectbox("Bölge", label_encoders['region'].classes_)
education = st.selectbox("Eğitim Durumu", label_encoders['education'].classes_)
gender = st.selectbox("Cinsiyet", label_encoders['gender'].classes_)
recruitment_channel = st.selectbox("İşe Alım Kanalı", label_encoders['recruitment_channel'].classes_)
no_of_trainings = st.number_input("Eğitim Sayısı", min_value=1, max_value=10, value=1, step=1)
age = st.number_input("Yaş", min_value=18, max_value=60, value=30, step=1)
previous_year_rating = st.number_input("Önceki Yıl Performans Notu", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
length_of_service = st.number_input("Hizmet Süresi (Yıl)", min_value=1, max_value=40, value=5, step=1)
kpi_met = st.selectbox("KPI >80% Karşılandı mı?", [0, 1])
awards_won = st.selectbox("Ödül Kazandı mı?", [0, 1])
avg_training_score = st.number_input("Ortalama Eğitim Skoru", min_value=0, max_value=100, value=50, step=1)

# Girdileri dönüştürme
encoded_inputs = [
    label_encoders['department'].transform([department])[0],
    label_encoders['region'].transform([region])[0],
    label_encoders['education'].transform([education])[0],
    label_encoders['gender'].transform([gender])[0],
    label_encoders['recruitment_channel'].transform([recruitment_channel])[0],
    no_of_trainings,
    age,
    previous_year_rating,
    length_of_service,
    kpi_met,
    awards_won,
    avg_training_score
]

# Tahmin yapma
if st.button("Terfi Durumunu Tahmin Et"):
    prediction = model.predict([encoded_inputs])
    result = "Terfi Edildi" if prediction[0] == 1 else "Terfi Edilmedi"
    st.write("Tahmin Sonucu:", result)
