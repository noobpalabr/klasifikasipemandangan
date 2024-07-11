from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='templates')

# Memuat data model punya renal
model = load_model('pemandangan.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pemandangan', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    img_file = request.files['image']
    img_path = 'uploaded_image.jpg'
    img_file.save(img_path)

    # Preprocessing gambar yang di upload bolo
    img = image.load_img(img_path, target_size=(90, 90))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Memastikan gambar bener
    img_array = img_array / 255.0 

    # Mencatat Bentuk Input
    app.logger.info(f"Image shape: {img_array.shape}")

    # Membuat prediksi bolo
    try:
        prediction = model.predict(img_array)
        class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Adjust based on your model classes
        predicted_class = class_names[np.argmax(prediction)]
        return jsonify({"class": predicted_class})
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)