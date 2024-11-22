def preprocess_image(image, image_ori):
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, remove_small_holes
    from scipy.ndimage import binary_fill_holes
    import cv2
    from skimage import img_as_float

    # Jika input adalah NumPy array, konversi ke PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Konversi gambar ke array untuk proses deteksi
    image_array = np.array(image)
    if isinstance(image_ori, np.ndarray):
        image_ori = Image.fromarray(image_ori)

    # Konversi gambar ke array untuk proses deteksi
    image_array = np.array(image)
    image_array_ori = np.array(image_ori)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Muat model Haar Cascade untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika tidak ada wajah terdeteksi, kembalikan error
    if len(faces) == 0:
        raise ValueError("Tidak ada wajah yang terdeteksi pada gambar.")

    # Potong gambar sesuai area wajah yang terdeteksi
    x, y, w, h = faces[0]  # Ambil wajah pertama yang terdeteksi
    cropped_image = image_array[y:y+h, x:x+w]
    cropped_image_ori = image_array_ori[y:y+h, x:x+w]

    # Konversi kembali ke PIL Image untuk resize
    image = Image.fromarray(cropped_image)
    image_ori = Image.fromarray(cropped_image_ori)

    # Resize gambar dengan mempertahankan aspect ratio
    width, height = image.size
    if height > width:
        maxLength = height
        if maxLength >= 6:
            image = image.resize((int(width * 580 / height), 580))
            image_ori = image_ori.resize((int(width * 580 / height), 580))
    else:
        maxLength = width
        if maxLength >= 6:
            image = image.resize((580, int(height * 580 / width)))
            image_ori = image_ori.resize((580, int(height * 580 / width)))

    resized_image_array = np.array(image)
    resized_image_array_ori = np.array(image_ori)

    # Konversi ke HSV untuk clustering
    citrahsv = cv2.cvtColor(resized_image_array, cv2.COLOR_RGB2HSV)
    hs = citrahsv[:, :, :2].reshape(-1, 2)

    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=2)
    cluster_idx = kmeans.fit_predict(hs)
    piksel_labels = cluster_idx.reshape(citrahsv.shape[:2])

    # Pilih cluster dengan objek terbanyak
    cluster_count = np.bincount(cluster_idx)
    max_cluster_index = np.argmax(cluster_count)

    # Segmentasi wajah
    citra_rgb = np.copy(resized_image_array)
    citra_rgb[piksel_labels != max_cluster_index] = 0
    citra_gray = rgb2gray(citra_rgb)
    level = threshold_otsu(citra_gray)
    binary_mask = citra_gray > level
    binary_mask = binary_mask.astype(bool)
    binary_mask = binary_fill_holes(binary_mask)
    binary_mask = binary_closing(binary_mask)
    binary_mask = remove_small_holes(binary_mask, area_threshold=100)

    citra_imfil = np.zeros_like(citra_rgb)
    citra_imfil[binary_mask] = citra_rgb[binary_mask]

    # Gabor filter
    citra_imfil_gray = rgb2gray(citra_imfil)
    citra_imfil_float = img_as_float(citra_imfil_gray)
    thetas = np.linspace(0, np.pi, 8)
    combined_output = np.zeros_like(citra_imfil_float, dtype=np.float32)

    for theta in thetas:
        gabor_kernel = cv2.getGaborKernel((12, 12), 2.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_output = cv2.filter2D(citra_imfil_float.astype(np.float32), cv2.CV_32F, gabor_kernel)
        combined_output += gabor_output

    combined_output = cv2.normalize(combined_output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    combined_output_enhanced = cv2.equalizeHist((combined_output * 255).astype(np.uint8))

    cv2.imshow('Preproccess', combined_output_enhanced)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows()
    cv2.imshow('Preproccess ori', resized_image_array_ori)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows()
    return combined_output_enhanced, resized_image_array_ori
