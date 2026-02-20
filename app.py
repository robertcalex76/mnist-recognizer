#!/usr/bin/env python3
"""Flask web server for MNIST digit recognition."""

import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from model import MNISTNet, preprocess_image

app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)

try:
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device, weights_only=True))
    model.eval()
    print('Model loaded successfully')
except FileNotFoundError:
    print('Warning: mnist_model.pth not found. Run train.py first.')


@app.route('/')
def index():
    """Serve the drawing interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Accept canvas image data and return prediction."""
    try:
        # Get base64 image data from request
        data = request.get_json()
        image_data = data.get('image', '')

        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess and predict
        tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_digit = output.argmax(dim=1).item()
            confidence = probabilities[predicted_digit].item()

        # Get all probabilities for display
        all_probs = {str(i): round(probabilities[i].item() * 100, 1)
                     for i in range(10)}

        return jsonify({
            'success': True,
            'prediction': predicted_digit,
            'confidence': round(confidence * 100, 1),
            'probabilities': all_probs
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
