

def prediction(image_results):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from PIL import Image
    from sklearn.cluster import KMeans
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, remove_small_holes
    from scipy.ndimage import binary_fill_holes
    import cv2
    from skimage import img_as_float

    try:
        # Memuat model dari file
        model_loaded = joblib.load('logistic_regression_model.pkl')
        print("Model berhasil dimuat.")

        # Memuat scaler dari file
        scaler = joblib.load('scaler_model.pkl')
        print("Scaler berhasil dimuat.")

        # Membuat DataFrame dari image_results
        data_baru = pd.DataFrame({
            'pixel_count_label_dahi': [image_results['pixel_count_label_dahi']],
            'pixel_count_label_tengah': [image_results['pixel_count_label_tengah']],
            'pixel_count_label_mata_kiri': [image_results['pixel_count_label_mata_kiri']],
            'pixel_count_label_mata_kanan': [image_results['pixel_count_label_mata_kanan']],
            'pixel_count_label_kantung_kiri': [image_results['pixel_count_label_kantung_kiri']],
            'pixel_count_label_kantung_kanan': [image_results['pixel_count_label_kantung_kanan']],
            'total_pixel_count': [image_results['total_pixel_count']],
        })

        # Transformasi data dengan scaler
        data_baru_scaled = scaler.transform(data_baru)

        # Prediksi dengan probabilitas
        probabilitas = model_loaded.predict_proba(data_baru_scaled)

        # Confidence untuk prediksi
        confidence = max(probabilitas[0])  # Probabilitas tertinggi

        # Prediksi kelas
        prediksi_baru = model_loaded.predict(data_baru_scaled)

        # Output hasil prediksi
        print(f"Probabilitas untuk setiap kelas: {probabilitas}")
        print(f"Confidence untuk prediksi: {confidence:.4f}")
        print(f"Hasil Prediksi untuk Data Baru: {prediksi_baru[0]}")

        return {
            "prediction": prediksi_baru[0],  # Kelas yang diprediksi
            "confidence": confidence,       # Tingkat keyakinan
            "probabilities": probabilitas[0].tolist()  # Probabilitas untuk setiap kelas
        }

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": "File model atau scaler tidak ditemukan."}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Terjadi kesalahan saat melakukan prediksi."}
