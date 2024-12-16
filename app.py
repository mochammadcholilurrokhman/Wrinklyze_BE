import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import requests
import cv2
from image_preprocessing import preprocess_image
from feature_extraction import feature_extraction
from prediction import prediction
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "mobile-app-wrinklyze.firebasestorage.app"
})

db = firestore.client()
bucket = storage.bucket()

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400

        # Download the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        # Save the image locally
        file_path = './uploads/temp_image.jpg'
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Load and preprocess the image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": "Failed to read the image."}), 400

        # Call preprocess_image and handle potential ValueError
        try:
            processed_image, image_ori = preprocess_image(img, img)
        except ValueError as e:
            return jsonify({"error": str(e), "face_detected": False}), 400  # Return error if no face is detected

        date = datetime.now()
        formatted_datetime = date.strftime("%Y-%m-%d %H:%M:%S")

        # Extract features from the image
        fitur = feature_extraction(processed_image, image_ori)

        # Make a prediction
        prediksi_result = prediction(fitur)
        title = f"Wrinklyze {formatted_datetime}"
        
        # Return prediction result in JSON format
        response = {
            "prediction": prediksi_result["prediction"],
            "confidence": prediksi_result["confidence"],
            "probabilities": prediksi_result["probabilities"],
            "title": title,
            "face_detected": True  # Indicate that a face was detected
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    
    app.run(debug=True, host='0.0.0.0', port=5000)



# import joblib
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify
# import firebase_admin
# from firebase_admin import credentials, firestore, storage
# import os
# from werkzeug.utils import secure_filename
# import cv2
# from image_preprocessing import preprocess_image
# from feature_extraction import feature_extraction
# from prediction import prediction

# app = Flask(__name__)

# # Initialize Firebase
# cred = credentials.Certificate("firebase_credentials.json")
# firebase_admin.initialize_app(cred, {
#     "storageBucket": "mobile-app-wrinklyze.firebasestorage.app"
# })

# db = firestore.client()
# bucket = storage.bucket()

# @app.route('/upload_file', methods=['POST'])
# def upload_file():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part in the request"}), 400

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify({"error": "No file selected"}), 400

#         # Check file format
#         if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             return jsonify({"error": "Invalid file format. Only .png, .jpg, and .jpeg are allowed."}), 400

#         in_memory_file = file.read()

#         # Convert byte data to image
#         npimg = np.frombuffer(in_memory_file, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Preprocess the image
#         processed_image, image_ori = preprocess_image(img, img)

#         # Extract features from the image
#         fitur = feature_extraction(processed_image, image_ori)

#         # Make a prediction
#         prediksi_result = prediction(fitur)

#         # Return prediction result in JSON format
#         response = {
#             "prediction": prediksi_result["prediction"],
#             "confidence": prediksi_result["confidence"],
#             "probabilities": prediksi_result["probabilities"]
#         }

#         return jsonify(response), 200

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)