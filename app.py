import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from werkzeug.utils import secure_filename
import cv2
from image_preprocessing import preprocess_image
from feature_extraction import feature_extraction
from prediction import prediction

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "mobile-app-wrinklyze.appspot.com"
})

db = firestore.client()
bucket = storage.bucket()

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check file format
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Only .png, .jpg, and .jpeg are allowed."}), 400

        in_memory_file = file.read()

        # Convert byte data to image
        npimg = np.fromstring(in_memory_file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_image, image_ori = preprocess_image(img, img)

        # Extract features from the image
        fitur = feature_extraction(processed_image, image_ori)

        # Make a prediction
        prediksi_result = prediction(fitur)

        # Return prediction result in JSON format
        response = {
            "prediction": prediksi_result["prediction"],
            "confidence": prediksi_result["confidence"],
            "probabilities": prediksi_result["probabilities"]
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
