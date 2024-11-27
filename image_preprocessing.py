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
    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(image_array_ori, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Konversi kembali ke PIL Image untuk resize
    image = Image.fromarray(image_array_ori)
    
    # Menampilkan gambar dengan kotak deteksi wajah
    cv2.imshow('Detected Faces', image_array_ori)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows()

    # Ambil wajah pertama yang terdeteksi
    x, y, w, h = faces[0]
    cropped_image = image_array[y:y+h, x:x+w]
    cropped_image_ori = image_array_ori[y:y+h, x:x+w]

    # Konversi kembali ke PIL Image untuk resize
    # Tentukan ukuran target (misalnya, panjang sisi maksimal 224)
    target_size = 580

    # Hitung rasio aspect gambar asli
    height, width = cropped_image.shape[:2]
    aspect_ratio = width / height

    # Tentukan dimensi baru dengan mempertahankan aspect ratio
    if aspect_ratio > 1:
        # Lebar lebih besar dari tinggi (portrait)
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Tinggi lebih besar atau sama dengan lebar (landscape)
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Resize gambar dengan ukuran yang dihitung
    resized_face = cv2.resize(cropped_image, (new_width, new_height))
    resized_face_asli = cv2.resize(cropped_image_ori, (new_width, new_height))
    
    cv2.imshow('Resized Face', resized_face)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows()
    cv2.imshow('Resized Face Asli', resized_face_asli)
    cv2.waitKey(0)  # Tunggu hingga tombol ditekan
    cv2.destroyAllWindows()
    return resized_face.astype(np.uint8), resized_face_asli.astype(np.uint8)