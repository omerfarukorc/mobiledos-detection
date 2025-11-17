# DoS Anomaly Detection in Mobile Network Traffic

Machine Learning tabanlı DoS/DDoS anomali tespit sistemi. CICIDS2017 dataset'i kullanılarak geliştirilmiştir.

## 📋 Proje Özeti

Bu proje, mobil ağ trafiğinde DoS/DDoS saldırılarını tespit etmek için **Unsupervised Learning (Isolation Forest)** ve **Supervised Learning (XGBoost, Random Forest, LightGBM)** yöntemlerini karşılaştırır.

## 🎯 Performans Sonuçları

### Unsupervised Learning (Isolation Forest)
- **DoS Detection Rate:** 88.45% ✅
- **Precision:** 93.28% ✅
- **False Alarm Rate:** 46.51% (Güvenlik için kabul edilebilir trade-off)
- **Missed Attacks:** 11.55% (3,851 / 33,333)

### Supervised Learning (XGBoost)
- **DoS Detection Rate:** 99.80% ✅
- **Accuracy:** 99.95% ✅
- **Precision:** 100.00% ✅
- **False Alarm Rate:** 0.20% ✅

## 🚀 Kullanım

### Gereksinimler
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm
```

### Unsupervised Learning (Isolation Forest)
```bash
python windows_dos_detection.py
```

### Supervised Learning (XGBoost, Random Forest, LightGBM)
```bash
python windows_dos_detection_supervised.py
```

## 📊 Veri Seti

**CICIDS2017** - Canadian Institute for Cybersecurity
- Normal Traffic: 529,918 flows
- DoS/DDoS Attacks: 380,688 flows
- Features: 77 numeric features

### Veri Seti İndirme
Veri setini [CICIDS2017 Official Website](https://www.unb.ca/cic/datasets/ids-2017.html) adresinden indirebilirsiniz.

Gerekli dosyalar:
- `Monday-WorkingHours.pcap_ISCX.csv` (Normal traffic)
- `Wednesday-workingHours.pcap_ISCX.csv` (DoS attacks)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS attacks)

İndirilen dosyaları `dataset/` klasörüne yerleştirin.

## 📈 Görselleştirmeler

Proje çalıştırıldığında `visualizations/` klasöründe şu grafikler oluşturulur:
- Confusion Matrix
- Performance Metrics
- Score Distribution
- Detection Rates
- Feature Importance
- Summary Report

## 📄 Raporlar

- **PROJE_RAPORU.md**: Detaylı proje raporu (Unsupervised + Supervised karşılaştırması)
- **PROJE_KARSILASTIRMALI_RAPOR.md**: Karşılaştırmalı performans analizi
- **Feature_Importance_Full.csv**: Feature importance değerleri

## 🔬 Metodoloji

### Isolation Forest (Unsupervised)
- **Avantajlar:**
  - Etiketli veriye ihtiyaç duymaz
  - Yeni/bilinmeyen saldırı türlerini tespit edebilir
  - Sadece normal trafik ile eğitilebilir
  
- **Dezavantajlar:**
  - Daha yüksek false alarm oranı
  - Supervised learning'e göre daha düşük accuracy

### XGBoost (Supervised)
- **Avantajlar:**
  - Çok yüksek doğruluk (%99.95)
  - Çok düşük false alarm (%0.20)
  - Bilinen saldırıları mükemmel tespit eder
  
- **Dezavantajlar:**
  - Etiketli veriye ihtiyaç duyar
  - Yeni saldırı türlerini tespit edemeyebilir

## 🎓 Akademik Kullanım

Bu proje akademik çalışmalar için geliştirilmiştir. Kullanırken lütfen kaynak gösterin.

## 📞 İletişim

Sorularınız için issue açabilirsiniz.

---

**Geliştirme Tarihi:** 2025  
**Dataset:** CICIDS2017  
**Yöntemler:** Isolation Forest, XGBoost, Random Forest, LightGBM

