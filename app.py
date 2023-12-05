import streamlit as app
import numpy as np
from PIL import ImageOps
import tensorflow as tf

# Model loading
trained_model = tf.keras.models.load_model('metalcastdetector.h5')

# Image preprocessing
def prepare_image(file):
    image = ImageOps.open(file)
    resized_image = image.resize((224, 224))
    image_to_array = np.array(resized_image) / 255.0
    expanded_array = np.expand_dims(image_to_array, axis=0)
    return expanded_array

# Defect detection in metal
def detect_defect(image_file):
    try:
        prepared_img = prepare_image(image_file)
        defect_prediction = trained_model.predict(prepared_img)
        return defect_prediction
    except Exception as error:
        app.error("Invalid Image. Try an image of metal casting.")
        return np.array([[1.0]])

def run_app():
    app.title("Metal Defect Detection Tool")

    user_uploaded_file = app.file_uploader("Upload your metal cast image", type=["jpg", "jpeg", "png"])

    if user_uploaded_file is not None:
        image_file_path = f"uploads/{user_uploaded_file.name}"
        with open(image_file_path, "wb") as file:
            file.write(user_uploaded_file.getbuffer())

        app.image(user_uploaded_file, caption="Your Image", use_column_width=True)

        if app.button("Check for Defects"):
            defect_result = detect_defect(image_file_path)
            outcome = "Defective" if defect_result[0][0] >= 0.5 else "Normal"
            app.write(f"Analysis Outcome: {outcome} (Value: {defect_result[0][0]:.2f})")

if __name__ == "__main__":
    run_app()
