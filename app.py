from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Inisialisasi Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Path ke file kredensial Firebase
firebase_admin.initialize_app(cred, {
    "storageBucket": "mobile-app-wrinklyze.appspot.com"  # Ganti dengan nama bucket
})

db = firestore.client()
bucket = storage.bucket()

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Simpan file sementara
        filename = secure_filename(file.filename)
        file_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        file.save(file_path)

        # Unggah file ke Firebase Storage
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_filename(file_path)
        blob.make_public()

        # Hapus file sementara
        os.remove(file_path)

        # Simpan metadata ke Firestore
        file_data = {
            "file_name": filename,
            "file_url": blob.public_url,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        doc_ref = db.collection("uploads").add(file_data)  # Ganti "uploads" dengan koleksi Anda

        return jsonify({
            "message": "File uploaded successfully",
            "file_url": blob.public_url,
            "document_id": doc_ref[1].id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Mengizinkan koneksi dari jaringan lokal


