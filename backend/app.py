from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
from inference_private import predict_private

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_flat = np.array(data['image']).reshape((28, 28, 1))
    pred = predict_private(image_flat)
    return jsonify({'prediction': pred})

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    app.run(debug=True)
