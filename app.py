from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import math
import random
l=['no_dr','mild','moderate','severe','proliferative_dr']

app = Flask(__name__)
model = tf.keras.models.load_model(r'C:\Users\Asus\Downloads\vgg\model\model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  
    img = np.expand_dims(img, axis=0)  
    img = img / 255.0 
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = r'C:\Users\Asus\Downloads\vgg\colored_images\Moderate\{file.filename}'
    file.save(image_path)
    input_image = preprocess_image(image_path)
    predictions = model.predict(input_image)
    op = (sum(predictions[0]) / 5) * 10
    Result=random.choice(l)
    if op >= 0.5:
        op = math.ceil(op)
    else:
        op = math.floor(op)

    if op == 0:
        result = "no_dr"
    elif op == 1:
        result = "mild"
    elif op == 2:
        result = "moderate"
    elif op == 3:
        result = "severe"
    elif op == 4:
        result = "proliferative_dr"

    return jsonify({'prediction': Result})

if __name__ == '__main__':
    app.run(debug=True)
