# SleepWell AI Analiz Sistemi

Bu proje, yasam tarzi verilerini kullanarak kisilerin Uyku Bozuklugu Riskini ve Bilisel Performans Skorunu tahmin eden uctan uca bir Makine Ogrenmesi uygulamasidir.

## Proje Hakkinda
Veri on islemeden model egitimine, API gelistirmeden Dockerize edilmesine kadar tum yazilim sureclerini kapsar.
* Modeller: LightGBM tabanli Siniflandirma ve Regresyon modelleri.
* Teknolojiler: Python, FastAPI, Scikit-learn, Docker, Tailwind CSS.
* Ozellikler: Otomatik BMI hesaplama, coklu model tahmini ve modern web arayuzu.

## Kurulum ve Calistirma (Docker)
Projeyi calistirmak icin Docker gereklidir. Terminalinizde sirasiyla asagidaki adimlari uygulayin:

1. Imaji Olusturun:
docker build -t sleepwell-app .

2. Konteyneri Baslatin:
docker run -p 8000:8000 sleepwell-app

Uygulama basladiktan sonra tarayicinizdan http://localhost:8000 adresine giderek arayuze ulasabilirsiniz.

## Dosya Yapisi
* app.py: FastAPI backend ve model entegrasyonu.
* models/: Egitilmis model dosyalari (.joblib).
* templates/: Web arayuzu (index.html).
* Dockerfile: Konteyner konfigurasyonu.

## Model Performansı

### Uyku Bozukluğu Riski Modeli (Sınıflandırma)
* **Doğruluk (Accuracy):** %94.1

### Bilişsel Performans Modeli (Regresyon)
* **R2 Skoru:** 0.932
* **RMSE:** 5.76
* **MAE:** 4.61