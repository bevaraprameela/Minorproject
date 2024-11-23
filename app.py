from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained machine learning model
model = load_model('model.h5')

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/')
def index():
    return render_template('tryit.htm')

@app.route('/predict', methods=['POST'])
def predict():
    a=['W180','W320','W400']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess the image to the format your model expects
        img_array = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Extract label and confidence from prediction
        label = np.argmax(prediction, axis=-1).item()
        confidence = np.max(prediction, axis=-1).item()
        
        return jsonify({"label": a[label]})

def preprocess_image(image):
    # Example preprocessing (adjust as needed for your model)
    image = image.resize((224, 224))  # Resize to expected input size
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

if __name__ == '__main__':
    app.run(debug=True, port=5001)

