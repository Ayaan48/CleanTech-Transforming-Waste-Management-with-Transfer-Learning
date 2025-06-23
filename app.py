from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("healthy_vs_rotten.h5")  # Replace with your model filename if needed

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ['Biodegradable', 'Recyclable', 'Trash']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return classes[predicted_class]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    prediction = model_predict(file_path)

    return render_template("index.html", prediction=prediction, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
