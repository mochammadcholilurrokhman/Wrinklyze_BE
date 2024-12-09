import joblib
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

# Set maximum content length for uploads (16 MB)
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

        processed_image, image_ori = preprocess_image(img, img)
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
            "title": title
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
