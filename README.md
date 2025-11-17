# DoS Anomaly Detection in Mobile Network Traffic

Machine Learning tabanlı DoS/DDoS saldırı tespit sistemi. CICIDS2017 veri seti kullanılarak geliştirildi.

## Proje Hakkında

Bu çalışmada mobil ağ trafiğinde DoS/DDoS saldırılarını tespit etmek için iki farklı yaklaşım kullanıldı:
- Unsupervised Learning (Isolation Forest)
- Supervised Learning (XGBoost, Random Forest, LightGBM)

## Sonuçlar

### Isolation Forest
- DoS Detection Rate: 88.45%
- Precision: 93.28%
- False Alarm Rate: 46.51%
- Missed Attacks: 11.55%

### XGBoost
- DoS Detection Rate: 99.80%
- Accuracy: 99.95%
- Precision: 100.00%
- False Alarm Rate: 0.20%

## Kurulum

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm
```

## Kullanım

Isolation Forest:
```bash
python windows_dos_detection.py
```

XGBoost, Random Forest, LightGBM:
```bash
python windows_dos_detection_supervised.py
```

## Veri Seti

CICIDS2017 - Canadian Institute for Cybersecurity
- Normal Traffic: 529,918 flows
- DoS/DDoS Attacks: 380,688 flows
- Features: 77 numeric features

Veri seti [buradan](https://www.unb.ca/cic/datasets/ids-2017.html) indirilebilir.

Gerekli dosyalar:
- Monday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

İndirilen dosyaları `dataset/` klasörüne yerleştirin.

## Çıktılar

Programlar çalışınca `visualizations/` klasöründe grafikler oluşur:
- Confusion Matrix
- Performance Metrics
- Score Distribution
- Detection Rates
- Feature Importance

Detaylı raporlar:
- PROJE_RAPORU.md
- PROJE_KARSILASTIRMALI_RAPOR.md
- Feature_Importance_Full.csv

## Yöntemler

**Isolation Forest:** Etiketli veriye ihtiyaç duymaz, sadece normal trafik ile eğitilir. Yeni saldırı türlerini tespit edebilir ancak false alarm oranı daha yüksek.

**XGBoost:** Hem normal hem saldırı örnekleri kullanır. Yüksek doğruluk sağlar ancak sadece bilinen saldırı türlerini tespit eder.
