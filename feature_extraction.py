

def feature_extraction(resized_face, resized_face_asli):
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
    import dlib
    from imutils import face_utils
    # Load the image
    # Initialize dlib's face detector and create a facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")  # Download the model from dlib's website
    # Detect faces in the resized_face image

    resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    resized_face_asli = cv2.cvtColor(resized_face_asli, cv2.COLOR_BGR2RGB)
    faces = detector(resized_face)
    # Get the image dimensions (height, width)
    height, width = resized_face.shape

    # Iterate over each face found in the image
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(resized_face, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Function to clip coordinates to be within the image boundaries
        def clip_coordinates(top, bottom, left, right):
            top = max(0, top)  # Ensure top is not less than 0
            bottom = min(height, bottom)  # Ensure bottom is not greater than the image height
            left = max(0, left)  # Ensure left is not less than 0
            right = min(width, right)  # Ensure right is not greater than the image width
            return top, bottom, left, right

        # Dahi (forehead)
        top = landmarks[68][1]
        bottom = landmarks[20][1]
        left = landmarks[75][0]
        right = landmarks[74][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraDahi = np.zeros_like(resized_face, dtype=np.float32)
        citraDahi[top:bottom, left:right] = resized_face[top:bottom, left:right]
        
        # Mata Tengah (middle of the eyes)
        top = landmarks[21][1]
        bottom = landmarks[28][1]
        left = landmarks[39][0]
        right = landmarks[42][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraMataT = np.zeros_like(resized_face, dtype=np.float32)
        citraMataT[top:bottom, left:right] = resized_face[top:bottom, left:right]

        # Mata Kiri (left eye)
        top = landmarks[77][1]
        bottom = landmarks[0][1]
        left = landmarks[0][0]
        right = landmarks[18][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraMataKr = np.zeros_like(resized_face, dtype=np.float32)
        citraMataKr[top:bottom, left:right] = resized_face[top:bottom, left:right]

        # Mata Kanan (right eye)
        top = landmarks[78][1]
        bottom = landmarks[16][1]
        left = landmarks[25][0]
        right = landmarks[16][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraMataKn = np.zeros_like(resized_face, dtype=np.float32)
        citraMataKn[top:bottom, left:right] = resized_face[top:bottom, left:right]

        # Kantung Mata Kiri (left eye bags)
        top = landmarks[41][1]
        bottom = landmarks[1][1]
        left = landmarks[1][0]
        right = landmarks[31][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraKantungKr = np.zeros_like(resized_face, dtype=np.float32)
        citraKantungKr[top:bottom, left:right] = resized_face[top:bottom, left:right]

        # Kantung Mata Kanan (right eye bags)
        top = landmarks[46][1]
        bottom = landmarks[15][1]
        left = landmarks[35][0]
        right = landmarks[15][0]
        top, bottom, left, right = clip_coordinates(top, bottom, left, right)
        
        citraKantungKn = np.zeros_like(resized_face, dtype=np.float32)
        citraKantungKn[top:bottom, left:right] = resized_face[top:bottom, left:right]

        # ==================== Canny Dahi ==================== #
        # 1. Deteksi tepi menggunakan Canny
        blurred_dahi = cv2.GaussianBlur(citraDahi, (5, 5), 0)

        canny_dahi = canny(blurred_dahi, sigma=1.5, low_threshold=0.1, high_threshold=0.2)

        # 2. Konversi hasil Canny ke tipe boolean
        canny_dahi_bw = canny_dahi.astype(bool)
        bw_canny, num_canny = label(canny_dahi_bw, connectivity=2, return_num=True)

        # ==================== Eliminasi Kandidat Keriput Dahi - Canny ==================== #
        # 1. Eliminasi Berdasarkan Luas dan Bentuk
        blob_measurements = regionprops(bw_canny)
        area1_dahi = np.array([blob.area for blob in blob_measurements])

        # Filter untuk blob dengan area lebih dari 100
        index_blob = np.where(area1_dahi > 85)[0]
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
        index_area = np.where(area2_dahi < 100)[0]
        ambil_area_dahi = np.isin(labeled_blob_canny_dahi, index_area + 1)

        # Blob hasil filter kedua
        blob_bw_final = ambil_area_dahi > 0
        label_area_bw_dahi, number_area_bw = label(blob_bw_final, connectivity=2, return_num=True)

        # ==================== Canny Sisi Tengah ==================== #
        blurred_tengah = cv2.GaussianBlur(citraMataT, (5, 5), 0)

        # 1. Canny Sisi Tengah
        canny_sisi_tengah = canny(blurred_tengah, sigma=1.5, low_threshold=0.1, high_threshold=0.2)

        # 2. BW LABEL
        canny_sisi_tengah_bw = canny_sisi_tengah.astype(bool)
        bw_sisi_t, num_canny = label(canny_sisi_tengah_bw, connectivity=2, return_num=True)

        # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
        # 1. Eliminasi Berdasarkan Luas dan Bentuk (Sisi Tengah Mata)
        blob_measurements = regionprops(bw_sisi_t)
        area1_sisi_tengah = np.array([blob.area for blob in blob_measurements])

        # Filter untuk blob dengan area lebih dari 50
        index_blob = np.where(area1_sisi_tengah > 30)[0]
        ambil_blob = np.isin(bw_sisi_t, index_blob + 1)  # `+1` karena label mulai dari 1

        # Blob yang tersaring
        blob_bw = ambil_blob > 0
        labeled_blob_canny_sisi_tengah, number_of_blobs = label(blob_bw, connectivity=2, return_num=True)

        # 2. Eliminasi Area Luas
        blob_measurements_area = regionprops(labeled_blob_canny_sisi_tengah)
        area2_bt = np.array([blob.area for blob in blob_measurements_area])

        # Filter untuk area yang lebih kecil dari 30
        index_area = np.where(area2_bt < 40)[0]
        ambil_area_tengah = np.isin(labeled_blob_canny_sisi_tengah, index_area + 1)

        # Blob hasil filter kedua
        blob_bw_final = ambil_area_tengah > 0
        label_area_bw_tengah, number_area_bw = label(blob_bw_final, connectivity=2, return_num=True)
        
        # ==================== Canny Sisi Mata ==================== #
        # 1. Canny untuk sisi mata kiri dan kanan
        blurred_mata_kiri = cv2.GaussianBlur(citraMataKr, (5, 5), 0)
        blurred_mata_kanan = cv2.GaussianBlur(citraMataKn, (5, 5), 0)

        canny_mata_kiri = canny(blurred_mata_kiri, sigma=1.5, low_threshold=0.1, high_threshold=0.2)
        canny_mata_kanan = canny(blurred_mata_kanan, sigma=1.5, low_threshold=0.1, high_threshold=0.2)

        # 2. BW LABEL untuk sisi mata kiri dan kanan
        bw_canny_mata_kiri, num_canny_kiri = label(canny_mata_kiri, connectivity=2, return_num=True)
        bw_canny_mata_kanan, num_canny_kanan = label(canny_mata_kanan, connectivity=2, return_num=True)

        # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
        # 1. Eliminasi Berdasarkan Luas dan Bentuk untuk Sisi Mata Kiri
        blob_measurements_kiri = regionprops(bw_canny_mata_kiri)
        area1_mata_kiri = np.array([blob.area for blob in blob_measurements_kiri])

        # Filter untuk blob dengan area lebih dari 50
        index_blob_kiri = np.where(area1_mata_kiri > 1)[0]
        ambil_blob_kiri = np.isin(bw_canny_mata_kiri, index_blob_kiri + 1)

        # Blob yang tersaring untuk mata kiri
        blob_bw_kiri = ambil_blob_kiri > 0
        labeled_blob_canny_mata_kiri, number_of_blobs_kiri = label(blob_bw_kiri, connectivity=2, return_num=True)

        # 1. Eliminasi Berdasarkan Luas dan Bentuk untuk Sisi Mata Kanan
        blob_measurements_kanan = regionprops(bw_canny_mata_kanan)
        area1_mata_kanan = np.array([blob.area for blob in blob_measurements_kanan])

        # Filter untuk blob dengan area lebih dari 50
        index_blob_kanan = np.where(area1_mata_kanan > 1)[0]
        ambil_blob_kanan = np.isin(bw_canny_mata_kanan, index_blob_kanan + 1)

        # Blob yang tersaring untuk mata kanan
        blob_bw_kanan = ambil_blob_kanan > 0
        labeled_blob_canny_mata_kanan, number_of_blobs_kanan = label(blob_bw_kanan, connectivity=2, return_num=True)

        # 2. Eliminasi Berdasarkan Luas dan Bentuk Bingkai
        # a. Sisi Mata Kiri
        blob_measurements_area_kiri = regionprops(labeled_blob_canny_mata_kiri)
        area2_mata_kiri = np.array([blob.area for blob in blob_measurements_area_kiri])

        # Filter untuk area yang lebih kecil dari 30
        index_blob_area_kiri = np.where(area2_mata_kiri < 50)[0]
        ambil_blob_area_kiri = np.isin(labeled_blob_canny_mata_kiri, index_blob_area_kiri + 1)

        # Blob hasil filter kedua untuk mata kiri
        blob_bw_final_kiri = ambil_blob_area_kiri > 0
        labeled_area_bw_mata_kiri, number_area_bw_kiri = label(blob_bw_final_kiri, connectivity=2, return_num=True)

        # b. Sisi Mata Kanan
        blob_measurements_area_kanan = regionprops(labeled_blob_canny_mata_kanan)
        area2_mata_kanan = np.array([blob.area for blob in blob_measurements_area_kanan])

        # Filter untuk area yang lebih kecil dari 30
        index_blob_area_kanan = np.where(area2_mata_kanan < 50)[0]
        ambil_blob_area_kanan = np.isin(labeled_blob_canny_mata_kanan, index_blob_area_kanan + 1)

        # Blob hasil filter kedua untuk mata kanan
        blob_bw_final_kanan = ambil_blob_area_kanan > 0
        labeled_area_bw_mata_kanan, number_area_bw_kanan = label(blob_bw_final_kanan, connectivity=2, return_num=True)

        # ==================== Canny Kantung Mata ==================== #
        # 1. Canny untuk kantung mata kiri dan kanan
        blurred_kantung_kiri = cv2.GaussianBlur(citraKantungKr, (5, 5), 0)
        blurred_kantung_kanan = cv2.GaussianBlur(citraKantungKn, (5, 5), 0)

        canny_kantung_kiri = canny(blurred_kantung_kiri, sigma=1.5, low_threshold=0.1, high_threshold=0.2)
        canny_kantung_kanan = canny(blurred_kantung_kanan, sigma=1.5, low_threshold=0.1, high_threshold=0.2)

        # 2. BW LABEL untuk kantung mata kiri dan kanan
        bw_canny_kantung_kiri, num_canny_kiri = label(canny_kantung_kiri, connectivity=2, return_num=True)
        bw_canny_kantung_kanan, num_canny_kanan = label(canny_kantung_kanan, connectivity=2, return_num=True)

        # ==================== Eliminasi Kandidat Keriput - Canny ==================== #
        # 1. Eliminasi Berdasarkan Luas dan Bentuk
        # a1. Kantung Kiri
        blob_measurements_kiri = regionprops(bw_canny_kantung_kiri)
        area1_kantung_kiri = np.array([blob.area for blob in blob_measurements_kiri])

        # Filter untuk blob dengan area lebih dari 35
        index_blob_kiri = np.where(area1_kantung_kiri > 55)[0]
        ambil_blob_kiri = np.isin(bw_canny_kantung_kiri, index_blob_kiri + 1)

        # Blob yang tersaring untuk kantung kiri
        blob_bw_kiri = ambil_blob_kiri > 0
        labeled_blob_canny_kantung_kiri, number_of_blobs_kiri = label(blob_bw_kiri, connectivity=2, return_num=True)

        # a2. Kantung Kiri, filter area lebih kecil dari 90
        blob_measurements_area_kiri = regionprops(labeled_blob_canny_kantung_kiri)
        area2_kantung_kiri = np.array([blob.area for blob in blob_measurements_area_kiri])

        index_blob_area_kiri = np.where(area2_kantung_kiri < 125)[0]
        ambil_blob_area_kiri = np.isin(labeled_blob_canny_kantung_kiri, index_blob_area_kiri + 1)

        blob_bw_final_kiri = ambil_blob_area_kiri > 0
        labeled_bw2_kantung_kiri, number_of_blobs_kiri = label(blob_bw_final_kiri, connectivity=2, return_num=True)

        # 2. Eliminasi Berdasarkan Luas dan Bentuk Bingkai
        # b1. Kantung Kanan, filter area lebih dari 20
        blob_measurements_kanan = regionprops(bw_canny_kantung_kanan)
        area1_kantung_kanan = np.array([blob.area for blob in blob_measurements_kanan])

        index_blob_kanan = np.where(area1_kantung_kanan > 20)[0]
        ambil_blob_kanan = np.isin(bw_canny_kantung_kanan, index_blob_kanan + 1)

        blob_bw_kanan = ambil_blob_kanan > 0
        labeled_blob_canny_kantung_kanan, number_of_blobs_kanan = label(blob_bw_kanan, connectivity=2, return_num=True)

        # b2. Kantung Kanan, filter area lebih dari 55
        blob_measurements_area_kanan = regionprops(labeled_blob_canny_kantung_kanan)
        area2_kantung_kanan = np.array([blob.area for blob in blob_measurements_area_kanan])

        index_blob_area_kanan = np.where(area2_kantung_kanan < 100)[0]
        ambil_blob_area_kanan = np.isin(labeled_blob_canny_kantung_kanan, index_blob_area_kanan + 1)

        blob_bw_final_area_kanan = ambil_blob_area_kanan > 0
        labeled_bw2_kantung_kanan, number_of_blobs_kanan = label(blob_bw_final_area_kanan, connectivity=2, return_num=True)


        # Pastikan citra asli dan hasil segmentasi diinisialisasi dengan benar
        hasil = resized_face_asli.copy()  # Salin citra asli untuk hasil marking

        # Dapatkan ukuran baris dan kolom dari citra
        bar, kol = resized_face_asli.shape[:2]

        # Daftar variabel label dari berbagai area yang akan ditandai
        labels_to_mark = [
            label_area_bw_dahi, label_area_bw_tengah, labeled_area_bw_mata_kiri, 
            labeled_area_bw_mata_kanan, labeled_bw2_kantung_kiri, labeled_bw2_kantung_kanan
        ]

        pixel_counts = []

        # Loop untuk menandai area yang sesuai pada citra hasil
        for image_idx, label in enumerate(labels_to_mark):
            count = 0
            label_resized = cv2.resize(label, (kol, bar), interpolation=cv2.INTER_NEAREST)

            # Counter untuk jumlah piksel bertanda pada label ini
            for i in range(bar):
                for j in range(kol):
                    if label_resized[i, j] > 0:
                        hasil[i, j, 0] = 0    # Kanal R (Merah) diatur ke 0
                        hasil[i, j, 1] = 0    # Kanal G (Hijau) diatur ke 0
                        hasil[i, j, 2] = 255  # Kanal B (Biru) diatur ke 255 (menandai dengan warna biru)
                        count += 1  # Tambahkan ke jumlah piksel bertanda untuk label ini
            pixel_counts.append(count)  # Simpan jumlah piksel bertanda untuk label ini

        # Menghitung total piksel bertanda untuk semua label
        total_pixel_count = sum(pixel_counts)

        image_results = {
            'image_idx': image_idx,
            'pixel_count_label_dahi': pixel_counts[0],
            'pixel_count_label_tengah': pixel_counts[1],
            'pixel_count_label_mata_kiri': pixel_counts[2],
            'pixel_count_label_mata_kanan': pixel_counts[3],
            'pixel_count_label_kantung_kiri': pixel_counts[4],
            'pixel_count_label_kantung_kanan': pixel_counts[5],
            'total_pixel_count': total_pixel_count,
        }


        # Ubah citra hasil menjadi tipe uint8 untuk ditampilkan
        hasil = hasil.astype(np.uint8)

        # Tampilkan citra asli dan citra hasil marking
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(resized_face_asli)
        plt.title('Citra Asli')

        plt.subplot(1, 2, 2)
        plt.imshow(hasil)
        plt.title('Ambil Keriput')

        plt.show()

        # Tampilkan hasil statistik
        print("\n=== Hasil Statistik ===")
        for key, value in image_results.items():
            print(f"{key}: {value}")
        
        return image_results