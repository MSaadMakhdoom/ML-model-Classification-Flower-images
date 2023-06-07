import cv2
import pickle5 as pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from io import BytesIO
from keras.models import load_model

app = Flask(__name__)

image_size = (128, 128)

loaded_model = load_model('model.h5')

# class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(image):
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    # Read and preprocess the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    predictions = loaded_model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = class_names[np.argmax(predictions[0])]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run()
