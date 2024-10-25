# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the MobileNetV2 model pre-trained on ImageNet dataset
model = MobileNetV2(weights='imagenet')

def recognize_image(img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")  # Open the image file
    img = img.resize((224, 224))  # Resize to 224x224 as required by MobileNetV2
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    # Predict the content of the image
    predictions = model.predict(img_array)

    # Decode and print the results
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}. {label} ({score * 100:.2f}%)")

# Path to the image file (you can replace this with any image path)
img_path = 'path_to_your_image.jpg'
recognize_image(img_path)
