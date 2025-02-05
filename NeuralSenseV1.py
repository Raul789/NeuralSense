import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import shutil
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import imutils
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, makedirs, remove

# ================================= NeuralSense V1.0 ===========================================

# ============================ Data Preparation & Preprocessing =================================

IMG_WIDTH = 224  # Set the desired width for the images
IMG_HEIGHT = 224 # Set the desired height for the images

def crop_brain_contour(image, plot=True):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding and morphological operations to remove noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and select the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find extreme points and crop the image
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    cropped_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    # Plot both images side by side
    if plot:
        plt.figure(figsize=(10, 5))

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Plot cropped image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')
        plt.axis('off')

        plt.show()

    return cropped_image

# Load data function
def load_data(dir_list, image_size):
    X, y = [], []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):
            image = cv2.imread(directory + '/' + filename)
            if image is not None:  # Ensure the image is loaded
                image = crop_brain_contour(image, plot=False)
                image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                image = image / 255.0
                X.append(image)
                y.append([1] if 'yes' in directory else [0])
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    return X, y

# Build CONVOLUTIONAL NEURAL NETWORK model
def build_model(input_shape):
    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = Input(input_shape) # shape=(?, 240, 240, 3)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X) # shape=(?, 59, 59, 32) 
    
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    
    # FLATTEN X 
    X = Flatten()(X) # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    
    return model

# Classify and save images to 'maligne' and 'benigne' folders
def classify_and_save_images(X, model, output_dir):
    makedirs(os.path.join(output_dir, 'maligne'), exist_ok=True)
    makedirs(os.path.join(output_dir, 'benigne'), exist_ok=True)

    for i, image in enumerate(X):
        image_input = np.expand_dims(image, axis=0)
        prediction = model.predict(image_input)
        class_label = 'maligne' if prediction[0][0] > 0.5 else 'benigne'
        output_path = os.path.join(output_dir, class_label, f'image_{i}.jpg')
        cv2.imwrite(output_path, (image * 255).astype(np.uint8))

# Count images in a directory
def count_images_in_directory(directory):
    return len([filename for filename in listdir(directory) if filename.endswith('.jpg') or filename.endswith('.png')])

# Clear contents of a directory
def clear_directory(directory):
    for filename in listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):  # Remove files
                remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

# ============================ SAVING FUNCTION FOR THE ACCURACY SCORES =============================

def save_accuracy_to_csv(file_path, accuracy_score, correct_count, incorrect_count):
    """ Save accuracy results to a CSV file with accuracy score, correct count, and incorrect count. """
    
    # Prepare the data to be saved
    data = {
        'Accuracy Score': [accuracy_score],
        'Correct_Count': [correct_count],
        'Incorrect_Count': [incorrect_count]
    }

    df = pd.DataFrame(data)

    # Print debug info
    print(f"Saving to CSV: Accuracy={accuracy_score:.2f}, Correct_Count={correct_count}, Incorrect_Count={incorrect_count}")

    # Append the DataFrame to the CSV file
    with open(file_path, 'a') as f:
        # If the file is new, write the header
        if f.tell() == 0:  # Check if file is empty
            f.write('Accuracy Score,Correct_Count,Incorrect_Count\n')
        
        # Write the data to CSV without the header
        df.to_csv(f, header=False, index=False)

    print(f"Data saved to {file_path} successfully.")


# ============================================== MAIN PROGRAM =========================================

# Other imports and code...

# ============================================== MAIN PROGRAM =========================================

if __name__ == "__main__":
    augmented_yes = 'yes'
    augmented_no = 'no'

    # Load data
    X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model((IMG_WIDTH, IMG_HEIGHT, 3))  # Assuming RGB images
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model for 1 or 10 epochs
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Process and classify test images after training
    predictions = []
    for i, image in enumerate(X_test):
        image_input = np.expand_dims(image, axis=0)
        prediction = model.predict(image_input)
        predicted_label = 1 if prediction[0][0] >= 0.5 else 0
        predictions.append(predicted_label)

    # Calculate accuracy
    if len(predictions) == 0 or len(y_test) == 0:
        print("No predictions made or no labels available.")
        exit()  # Exit if no predictions or labels are available
    
    accuracy = np.mean(np.array(predictions) == y_test.flatten())  # Ensure the shapes match
    accuracy_percentage = accuracy * 100
    correct_count = np.sum(np.array(predictions) == y_test.flatten())
    incorrect_count = len(y_test) - correct_count

    # Debugging output
    print(f"Correct Count: {correct_count}")
    print(f"Incorrect Count: {incorrect_count}")

    # Print final test accuracy
    print(f'Final Test Accuracy: {accuracy_percentage:.2f}%')

    # Plot accuracy as a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie([correct_count, incorrect_count], labels=['Correct', 'Incorrect'], colors=['green', 'red'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Model Prediction Accuracy')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    plt.show()

    # Save accuracy to CSV
    save_accuracy_to_csv('accuracy_scores.csv', accuracy_percentage, correct_count, incorrect_count)