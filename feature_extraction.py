

def feature_extraction(combined_output_enhanced, resized_image_array):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.cluster import KMeans
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, remove_small_holes
    from scipy.ndimage import binary_fill_holes
    from skimage.feature import canny
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects
    import cv2
    from skimage import img_as_float
    from scipy.stats import kurtosis, skew
    # Pastikan gambar sudah didefinisikan sebagai combined_output_enhanced
    # combined_output_enhanced = image_path  # Asumsikan gambar tunggal ada di index pertama
    cv2.imshow('Combine',combined_output_enhanced)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows() 

    cv2.imshow('Resized',resized_image_array)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows() 
    # Get the shape of the image
    bar, kol = combined_output_enhanced.shape

    # 1. DAHI
    citraDahi = np.zeros((bar, kol), dtype=np.float32)
    baris_dahi1 = round(bar * 10 / 100)
    baris_dahi2 = round(bar * 26 / 100)
    kolom_dahi1 = round(kol * 17 / 100)
    kolom_dahi2 = round(kol * 80 / 100)
    citraDahi[baris_dahi1:baris_dahi2, kolom_dahi1:kolom_dahi2] = combined_output_enhanced[baris_dahi1:baris_dahi2, kolom_dahi1:kolom_dahi2]

    # 2. Bagian Tengah Mata
    citraMataT = np.zeros((bar, kol), dtype=np.float32)
    baris_mataT1 = round(bar * 27 / 100)
    baris_mataT2 = round(bar * 37 / 100)
    kolom_mataT1 = round(kol * 42 / 100)
    kolom_mataT2 = round(kol * 56 / 100)
    citraMataT[baris_mataT1:baris_mataT2, kolom_mataT1:kolom_mataT2] = combined_output_enhanced[baris_mataT1:baris_mataT2, kolom_mataT1:kolom_mataT2]

    # 3. Mata Kiri
    citraMataKr = np.zeros((bar, kol), dtype=np.float32)
    baris_matakr1 = round(bar * 32 / 100)
    baris_matakr2 = round(bar * 42 / 100)
    kolom_matakr1 = round(kol * 3  / 100)
    kolom_matakr2 = round(kol * 18 / 100)
    citraMataKr[baris_matakr1:baris_matakr2, kolom_matakr1:kolom_matakr2] = combined_output_enhanced[baris_matakr1:baris_matakr2, kolom_matakr1:kolom_matakr2]

    # 4. Mata Kanan
    citraMataKn = np.zeros((bar, kol), dtype=np.float32)
    kolom_matakn1 = round(kol * 82 / 100)
    kolom_matakn2 = round(kol * 97 / 100)
    citraMataKn[baris_matakr1:baris_matakr2, kolom_matakn1:kolom_matakn2] = combined_output_enhanced[baris_matakr1:baris_matakr2, kolom_matakn1:kolom_matakn2]

    # 5. Kantung Mata Kiri
    citraKantungKr = np.zeros((bar, kol), dtype=np.float32)
    baris_kantungkr1 = round(bar * 43 / 100)
    baris_kantungkr2 = round(bar * 52 / 100)
    kolom_kantungkr1 = round(kol * 7 / 100)
    kolom_kantungkr2 = round(kol * 43 / 100)
    citraKantungKr[baris_kantungkr1:baris_kantungkr2, kolom_kantungkr1:kolom_kantungkr2] = combined_output_enhanced[baris_kantungkr1:baris_kantungkr2, kolom_kantungkr1:kolom_kantungkr2]

    # 6. Kantung Mata Kanan
    citraKantungKn = np.zeros((bar, kol), dtype=np.float32)
    kolom_kantungkn1 = round(kol * 57 / 100)
    kolom_kantungkn2 = round(kol * 93 / 100)
    citraKantungKn[baris_kantungkr1:baris_kantungkr2, kolom_kantungkn1:kolom_kantungkn2] = combined_output_enhanced[baris_kantungkr1:baris_kantungkr2, kolom_kantungkn1:kolom_kantungkn2]


    # Asumsikan citraDahi sudah didefinisikan
    # ==================== Canny Dahi ==================== #
    # 1. Deteksi tepi menggunakan Canny
    canny_dahi = canny(citraDahi, sigma=1.0, low_threshold=0.08, high_threshold=0.16)

    # 2. Konversi hasil Canny ke tipe boolean
    canny_dahi_bw = canny_dahi.astype(bool)
    bw_canny, num_canny = label(canny_dahi_bw, connectivity=2, return_num=True)

    # ==================== Eliminasi Kandidat Keriput Dahi - Canny ==================== #
    # 1. Eliminasi Berdasarkan Luas dan Bentuk
    blob_measurements = regionprops(bw_canny)
    area1_dahi = np.array([blob.area for blob in blob_measurements])

    # Filter untuk blob dengan area lebih dari 100
    index_blob = np.where(area1_dahi > 100)[0]
    ambil_blob = np.isin(bw_canny, index_blob + 1)

    # Blob yang tersaring
    blob_bw = ambil_blob > 0
    labeled_blob_canny_dahi, number_of_blobs = label(blob_bw, connectivity=2, return_num=True)

    # 2. Eliminasi Berdasarkan Area yang Lebih Kecil dari 200
    blob_measurements_area = regionprops(labeled_blob_canny_dahi)
    area2_dahi = np.array([blob.area for blob in blob_measurements_area])

    mean_area = np.mean(area2_dahi)
    std_area = np.std(area2_dahi)

    # Filter untuk area yang lebih kecil dari 200
    index_area = np.where(area2_dahi < 200)[0]
    ambil_area_dahi = np.isin(labeled_blob_canny_dahi, index_area + 1)

    # Blob hasil filter kedua
    blob_bw_final = ambil_area_dahi > 0
    label_area_bw_dahi, number_area_bw = label(blob_bw_final, connectivity=2, return_num=True)


    # ==================== Canny Sisi Tengah ==================== #
    # 1. Canny Sisi Tengah
    canny_sisi_tengah = canny(citraMataT, sigma=1.0, low_threshold=0.04, high_threshold=0.8)

    # 2. BW LABEL
    canny_sisi_tengah_bw = canny_sisi_tengah.astype(bool)
    bw_sisi_t, num_canny = label(canny_sisi_tengah_bw, connectivity=2, return_num=True)

    # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
    # 1. Eliminasi Berdasarkan Luas dan Bentuk (Sisi Tengah Mata)
    blob_measurements = regionprops(bw_sisi_t)
    area1_sisi_tengah = np.array([blob.area for blob in blob_measurements])

    # Filter untuk blob dengan area lebih dari 50
    index_blob = np.where(area1_sisi_tengah > 50)[0]
    ambil_blob = np.isin(bw_sisi_t, index_blob + 1)  # `+1` karena label mulai dari 1

    # Blob yang tersaring
    blob_bw = ambil_blob > 0
    labeled_blob_canny_sisi_tengah, number_of_blobs = label(blob_bw, connectivity=2, return_num=True)

    # 2. Eliminasi Area Luas
    blob_measurements_area = regionprops(labeled_blob_canny_sisi_tengah)
    area2_bt = np.array([blob.area for blob in blob_measurements_area])

    # Filter untuk area yang lebih kecil dari 100
    index_area = np.where(area2_bt < 100)[0]
    ambil_area_tengah = np.isin(labeled_blob_canny_sisi_tengah, index_area + 1)

    # Blob hasil filter kedua
    blob_bw_final = ambil_area_tengah > 0
    label_area_bw_tengah, number_area_bw = label(blob_bw_final, connectivity=2, return_num=True)


    # ==================== Canny Sisi Mata ==================== #
    # 1. Canny untuk sisi mata kiri dan kanan
    canny_mata_kiri = canny(citraMataKr, sigma=1.0, low_threshold=0.01, high_threshold=0.2)
    canny_mata_kanan = canny(citraMataKn, sigma=1.0, low_threshold=0.01, high_threshold=0.2)

    # 2. BW LABEL untuk sisi mata kiri dan kanan
    bw_canny_mata_kiri, num_canny_kiri = label(canny_mata_kiri, connectivity=2, return_num=True)
    bw_canny_mata_kanan, num_canny_kanan = label(canny_mata_kanan, connectivity=2, return_num=True)

    # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
    # 1. Eliminasi Berdasarkan Luas dan Bentuk untuk Sisi Mata Kiri
    blob_measurements_kiri = regionprops(bw_canny_mata_kiri)
    area1_mata_kiri = np.array([blob.area for blob in blob_measurements_kiri])

    # Filter untuk blob dengan area lebih dari 50
    index_blob_kiri = np.where(area1_mata_kiri > 30)[0]
    ambil_blob_kiri = np.isin(bw_canny_mata_kiri, index_blob_kiri + 1)

    # Blob yang tersaring untuk mata kiri
    blob_bw_kiri = ambil_blob_kiri > 0
    labeled_blob_canny_mata_kiri, number_of_blobs_kiri = label(blob_bw_kiri, connectivity=2, return_num=True)

    # 1. Eliminasi Berdasarkan Luas dan Bentuk untuk Sisi Mata Kanan
    blob_measurements_kanan = regionprops(bw_canny_mata_kanan)
    area1_mata_kanan = np.array([blob.area for blob in blob_measurements_kanan])

    # Filter untuk blob dengan area lebih dari 50
    index_blob_kanan = np.where(area1_mata_kanan > 30)[0]
    ambil_blob_kanan = np.isin(bw_canny_mata_kanan, index_blob_kanan + 1)

    # Blob yang tersaring untuk mata kanan
    blob_bw_kanan = ambil_blob_kanan > 0
    labeled_blob_canny_mata_kanan, number_of_blobs_kanan = label(blob_bw_kanan, connectivity=2, return_num=True)

    # 2. Eliminasi Berdasarkan Luas dan Bentuk Bingkai
    # a. Sisi Mata Kiri
    blob_measurements_area_kiri = regionprops(labeled_blob_canny_mata_kiri)
    area2_mata_kiri = np.array([blob.area for blob in blob_measurements_area_kiri])

    # Filter untuk area yang lebih kecil dari 100
    index_blob_area_kiri = np.where(area2_mata_kiri < 100)[0]
    ambil_blob_area_kiri = np.isin(labeled_blob_canny_mata_kiri, index_blob_area_kiri + 1)

    # Blob hasil filter kedua untuk mata kiri
    blob_bw_final_kiri = ambil_blob_area_kiri > 0
    labeled_area_bw_mata_kiri, number_area_bw_kiri = label(blob_bw_final_kiri, connectivity=2, return_num=True)

    # b. Sisi Mata Kanan
    blob_measurements_area_kanan = regionprops(labeled_blob_canny_mata_kanan)
    area2_mata_kanan = np.array([blob.area for blob in blob_measurements_area_kanan])

    # Filter untuk area yang lebih kecil dari 100
    index_blob_area_kanan = np.where(area2_mata_kanan < 100)[0]
    ambil_blob_area_kanan = np.isin(labeled_blob_canny_mata_kanan, index_blob_area_kanan + 1)

    # Blob hasil filter kedua untuk mata kanan
    blob_bw_final_kanan = ambil_blob_area_kanan > 0
    labeled_area_bw_mata_kanan, number_area_bw_kanan = label(blob_bw_final_kanan, connectivity=2, return_num=True)


    # ==================== Canny Kantung Mata ==================== #
    # Loop untuk deteksi kantung mata kiri dan kanan

    # 1. Canny untuk kantung mata kiri dan kanan
    canny_kantung_kiri = canny(citraKantungKr, sigma=1.0, low_threshold=0.10, high_threshold=0.20)
    canny_kantung_kanan = canny(citraKantungKn, sigma=1.0, low_threshold=0.10, high_threshold=0.20)

    # 2. BW LABEL untuk kantung mata kiri dan kanan
    bw_canny_kantung_kiri, num_canny_kiri = label(canny_kantung_kiri, connectivity=2, return_num=True)
    bw_canny_kantung_kanan, num_canny_kanan = label(canny_kantung_kanan, connectivity=2, return_num=True)

    # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
    # 1. Eliminasi Berdasarkan Luas dan Bentuk untuk Kantung Mata Kiri
    blob_measurements_kantung_kiri = regionprops(bw_canny_kantung_kiri)
    area1_kantung_kiri = np.array([blob.area for blob in blob_measurements_kantung_kiri])

    # Filter untuk blob dengan area lebih dari 35
    index_blob_kantung_kiri = np.where(area1_kantung_kiri > 35)[0]
    ambil_blob_kantung_kiri = np.isin(bw_canny_kantung_kiri, index_blob_kantung_kiri + 1)

    # Blob yang tersaring untuk kantung kiri
    blob_bw_kantung_kiri = ambil_blob_kantung_kiri > 0
    labeled_blob_canny_kantung_kiri, number_of_blobs_kantung_kiri = label(blob_bw_kantung_kiri, connectivity=2, return_num=True)

    # 2. Eliminasi Berdasarkan Luas dan Bentuk Bingkai untuk Kantung Kiri
    blob_measurements_area_kantung_kiri = regionprops(labeled_blob_canny_kantung_kiri)
    area2_kantung_kiri = np.array([blob.area for blob in blob_measurements_area_kantung_kiri])

    index_blob_area_kantung_kiri = np.where(area2_kantung_kiri < 90)[0]
    ambil_blob_area_kantung_kiri = np.isin(labeled_blob_canny_kantung_kiri, index_blob_area_kantung_kiri + 1)

    blob_bw_final_kantung_kiri = ambil_blob_area_kantung_kiri > 0
    labeled_bw2_kantung_kiri, number_of_blobs_kantung_kiri = label(blob_bw_final_kantung_kiri, connectivity=2, return_num=True)

    # 3. Eliminasi untuk Kantung Mata Kanan
    blob_measurements_kantung_kanan = regionprops(bw_canny_kantung_kanan)
    area1_kantung_kanan = np.array([blob.area for blob in blob_measurements_kantung_kanan])

    index_blob_kantung_kanan = np.where(area1_kantung_kanan > 35)[0]
    ambil_blob_kantung_kanan = np.isin(bw_canny_kantung_kanan, index_blob_kantung_kanan + 1)

    blob_bw_kantung_kanan = ambil_blob_kantung_kanan > 0
    labeled_blob_canny_kantung_kanan, number_of_blobs_kantung_kanan = label(blob_bw_kantung_kanan, connectivity=2, return_num=True)

    # 4. Eliminasi Berdasarkan Luas dan Bentuk Bingkai untuk Kantung Kanan
    blob_measurements_area_kantung_kanan = regionprops(labeled_blob_canny_kantung_kanan)
    area2_kantung_kanan = np.array([blob.area for blob in blob_measurements_area_kantung_kanan])

    index_blob_area_kantung_kanan = np.where(area2_kantung_kanan > 55)[0]
    ambil_blob_area_kantung_kanan = np.isin(labeled_blob_canny_kantung_kanan, index_blob_area_kantung_kanan + 1)

    blob_bw_final_area_kantung_kanan = ambil_blob_area_kantung_kanan > 0
    labeled_bw_kantung_kanan, number_of_blobs_kantung_kanan = label(blob_bw_final_area_kantung_kanan, connectivity=2, return_num=True)

    # Filter area lebih kecil dari 140
    blob_measurements_final_area_kantung_kanan = regionprops(labeled_bw_kantung_kanan)
    area3_kantung_kanan = np.array([blob.area for blob in blob_measurements_final_area_kantung_kanan])

    index_blob_final_area_kantung_kanan = np.where(area3_kantung_kanan < 140)[0]
    ambil_blob_final_area_kantung_kanan = np.isin(labeled_bw_kantung_kanan, index_blob_final_area_kantung_kanan + 1)

    blob_bw_final_kanan = ambil_blob_final_area_kantung_kanan > 0
    labeled_bw3_kantung_kanan, number_of_blobs_kantung_kanan = label(blob_bw_final_kanan, connectivity=2, return_num=True)

    # Pastikan citra asli dan hasil segmentasi diinisialisasi dengan benar
    hasil = resized_image_array.copy()  # Salin citra asli untuk hasil marking

    # Dapatkan ukuran baris dan kolom dari citra
    bar, kol = resized_image_array.shape[:2]

    # Daftar variabel label dari berbagai area yang akan ditandai
    labels_to_mark = [
        label_area_bw_dahi, label_area_bw_tengah, labeled_area_bw_mata_kiri, 
        labeled_area_bw_mata_kanan, labeled_bw2_kantung_kiri, labeled_bw3_kantung_kanan
    ]

    pixel_counts = []
    image_features = []

    # Loop untuk menandai area yang sesuai pada citra hasil
    for image_idx, label in enumerate(labels_to_mark):
        count = 0
        label_resized = cv2.resize(label, (kol, bar), interpolation=cv2.INTER_NEAREST)

        # Counter untuk jumlah piksel bertanda pada label ini
        for i in range(bar):
            for j in range(kol):
                if label_resized[i, j] > 0:
                    hasil[i, j, 0] = 255    # Kanal R (Merah) diatur ke 0
                    hasil[i, j, 1] = 0    # Kanal G (Hijau) diatur ke 0
                    hasil[i, j, 2] = 0  # Kanal B (Biru) diatur ke 255 (menandai dengan warna biru)
                    count += 1  # Tambahkan ke jumlah piksel bertanda untuk label ini
        pixel_counts.append(count)  # Simpan jumlah piksel bertanda untuk label ini

    # Menghitung total piksel bertanda untuk semua label
    total_pixel_count = sum(pixel_counts)

    # Menghitung kepadatan piksel kerutan (total piksel bertanda dibagi total piksel area)
    pixel_density = total_pixel_count / (bar * kol)

    # Fitur tambahan untuk mendeteksi kerutan
    entropy_value = -np.sum(np.log2(np.histogram(hasil.flatten(), bins=256)[0] + 1e-5))

    # Fractal dimension (perhitungan kasar, dapat disesuaikan lebih lanjut)
    fractal_dimension = np.sum(np.abs(np.diff(hasil, axis=0)))

    # Menghitung gradien citra di kedua dimensi
    gradient_x = np.gradient(hasil, axis=1)  # Gradien di arah horizontal
    gradient_y = np.gradient(hasil, axis=0)  # Gradien di arah vertikal

    # Menghitung magnitudo gradien
    gradient_magnitude = np.mean(np.sqrt(gradient_x ** 2 + gradient_y ** 2))

    # Menghitung HOG (Histogram of Oriented Gradients) features
    hog_descriptor = cv2.HOGDescriptor()
    hog_features = hog_descriptor.compute(hasil)

    # Menghitung skewness dan kurtosis dari hasil gambar
    skewness_value = skew(hasil.flatten())
    kurtosis_value = kurtosis(hasil.flatten())

    image_results = {
        'image_idx': image_idx,
        'pixel_count_label_dahi': pixel_counts[0],
        'pixel_count_label_tengah': pixel_counts[1],
        'pixel_count_label_mata_kiri': pixel_counts[2],
        'pixel_count_label_mata_kanan': pixel_counts[3],
        'pixel_count_label_kantung_kiri': pixel_counts[4],
        'pixel_count_label_kantung_kanan': pixel_counts[5],
        'total_pixel_count': total_pixel_count,
        'pixel_density': pixel_density,
        'entropy_value': entropy_value,
        'fractal_dimension': fractal_dimension,
        'gradient_magnitude': gradient_magnitude,
        'hog_features': np.mean(hog_features),
        'skewness_value': skewness_value,
        'kurtosis_value': kurtosis_value
    }


    # Ubah citra hasil menjadi tipe uint8 untuk ditampilkan
    hasil = hasil.astype(np.uint8)
    cv2.imshow('Citra asli',resized_image_array)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows() 
    cv2.imshow('Ambil keriput',hasil)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows() 
    # Tampilkan hasil statistik
    print("\n=== Hasil Statistik ===")
    for key, value in image_results.items():
        print(f"{key}: {value}")
        
    return image_results



