# Prediksi Harga Properti dengan Model Advanced

## 📚 Overview

Proyek ini bertujuan untuk memprediksi harga properti berdasarkan data fitur seperti lokasi, luas bangunan, jumlah kamar, dan fasilitas. Menggunakan teknik Gradient Boosting dan Neural Network, proyek ini memberikan wawasan prediktif yang dapat digunakan dalam analisis pasar properti.

## 🎯 Tujuan

- Memprediksi harga properti berdasarkan fitur penting.
- Membandingkan performa Gradient Boosting dan Neural Network.

## 📂 Struktur Proyek

property-price-prediction/
├── data/
│ └── dataset_harga_properti.csv
├── notebooks/
│ └── property_price_analysis.ipynb  
├── results/
│ └── correlation_heatmap.png
│ └── gradient_boosting_model.pkl
│ └── neural_network_model.h5
│ └── prediction_vs_actual.png
│ └── price_distribution.png
├── scripts/
│ └── train_model.py
├── LICENSE
├── README.md  
├── requirements.txt

## 🛠️ Instalasi

1. Clone repository:

   ```bash
   git clone https://github.com/yourusername/property-price-prediction.git
   cd property-price-prediction

   ```

2. Install pustaka:
   pip install -r requirements.txt

3. Jalankan analisis:
   jupyter notebook notebooks/property_price_analysis.ipynb

## 📊 Hasil

││ Model ││ MAE ││ R² Score ││
││ Gradient Boosting ││ 18.75 ││ 0.99 ││
││ Neural Network ││ 48.84 ││ 0.93 ││

## ✍️ Author

- Septiyadi Kurnia Putra
