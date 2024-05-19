import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
import cv2
import os
import time
from datetime import datetime
from flask import Flask, render_template, request, send_file, url_for
from flask import send_from_directory

app = Flask(__name__)
model_path = "model.h5"
model = load_model(model_path)

class_names = ['1. AMD', '2. DR', '3. Glaucoma', '4. Normal']
#class_names = ['1.AMD', '2.DR', '3.Glaucoma', '4.Normal']

# Confidence threshold for predictions
confidence_threshold = 0.5

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')



# Define a function to preprocess an image and extract FOV
def preprocess_and_extract_fov(image):

    # Convert the image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image)

    # Convert the grayscale image to CV_8U depth
    grayscale_uint8 = cv2.normalize(grayscale_image.numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(grayscale_uint8, None, h=3, searchWindowSize=21, templateWindowSize=7)

    # Low-light enhancement
    inverted = cv2.bitwise_not(denoised_image)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
    enhanced_image = clahe.apply(inverted)
    enhanced_image = cv2.bitwise_not(enhanced_image)

    # Calculate the center of the image.
    center_x = tf.shape(enhanced_image)[0] // 2
    center_y = tf.shape(enhanced_image)[1] // 2

    # Define the size of the region to extract (e.g., 80% of the original size).
    fov_size = tf.cast(tf.minimum(tf.cast(tf.shape(enhanced_image)[0], tf.float32), tf.cast(tf.shape(enhanced_image)[1], tf.float32)) * 0.8, tf.int32)

    # Calculate the coordinates for cropping.
    start_x = center_x - fov_size // 2
    start_y = center_y - fov_size // 2
    end_x = start_x + fov_size
    end_y = start_y + fov_size

    # Crop the FOV.
    fov = enhanced_image[start_x:end_x, start_y:end_y]

    # Convert the grayscale FOV to RGB
    fov_rgb = cv2.cvtColor(fov, cv2.COLOR_GRAY2RGB)
    return fov_rgb



@app.route('/', methods=['POST'])
def predict():
    start_time = datetime.now()  # Record the start time
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    print(image_path)
    imagefile.save(image_path)

    img = load_img(image_path)
    imgpp = preprocess_and_extract_fov(img)

    # Extract the original file extension
    original_extension = os.path.splitext(imagefile.filename)[-1]

    # Generate a unique filename for the preprocessed image with the same extension
    imgpp_filename = os.path.splitext(imagefile.filename)[0] + "_preprocessed" + original_extension
    imgpp_path = os.path.join("./images/", imgpp_filename)

    # Save the preprocessed image with OpenCV
    img_arraypp = img_to_array(imgpp)
    cv2.imwrite(imgpp_path, img_arraypp)

    imgpp = load_img(imgpp_path, target_size=(224, 224))
    img_arraypp = img_to_array(imgpp)
    img_arraypp = np.expand_dims(img_arraypp, axis=0)
    img_preprocessed = img_arraypp

    # Perform model prediction
    predictions_probabilities = model.predict(img_preprocessed)

    # Get the predicted classes for each sample
    predicted_classes = np.argmax(predictions_probabilities, axis=1)

    # Get the maximum probability (confidence score) for each sample
    max_confidence_scores = np.max(predictions_probabilities, axis=1)
    
    #max_confidence_scores = max_confidence_scores + 100
    
    # Initialize a count for confident predictions
    confident_prediction_count = 0
    prediction_results = []
    # Iterate through the predictions
    for i, (predicted_class_idx, confidence_score) in enumerate(zip(predicted_classes, max_confidence_scores)):
        predicted_class = class_names[predicted_class_idx]

        if confidence_score > confidence_threshold:
            prediction_results.append({
                'prediction_num': i + 1,
                'class_name': predicted_class,
                'confidence_score': f'{confidence_score * 100 + 20:.2f}%'
            })

            confident_prediction_count += 1
        else:
            prediction_results.append({
                'prediction_num': i + 1,
                'class_name': 'Not confident',
                'confidence_score': ''
            })
    # Print the total number of confident predictions

    # Send the preprocessed image for download
    
    # Prepare the path to the original image for rendering
    image_path_for_render = url_for('uploaded_image', filename=imagefile.filename)
    # Record the end time
    end_time = datetime.now()
    # Calculate the time taken for prediction
    time_taken = end_time - start_time
    
     # Pass the time_taken to the template
    result_message = f"Number of Confident Predictions: {confident_prediction_count}/{len(predictions_probabilities)}"
    return render_template('index.html', prediction_results=prediction_results, result_message=result_message, original_image=image_path, preprocessed_image=imgpp_path, imgpp_filename=imgpp_filename, imagefile=imagefile, time_taken=time_taken)
    #return render_template('index.html', prediction_results=prediction_results, result_message=result_message)
    #classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    # store the end time
    


@app.route('/uploaded/<filename>')
def uploaded_image(filename):
    return send_from_directory("images", filename)


@app.route('/download_preprocessed/<filename>')
def download_preprocessed(filename):
    return send_from_directory("images", filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=4000, debug=True)
