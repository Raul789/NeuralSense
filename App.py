import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import shutil
import os
import csv
import json
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
from matplotlib import pyplot as plt


# ============================ Data Preparation & Preprocessing =================================

def crop_brain_contour(image, plot=False):
    if image is None:
        print("Error: Image not loaded properly.")
        return None

    try:
        # Convert the image to grayscale and blur it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image, then remove small noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours and grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            print("No contours found.")
            return None
        
        c = max(cnts, key=cv2.contourArea)

        # Find the extreme points and crop the image
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
            plt.title('Cropped Image')
            plt.axis('off')
            plt.show()

        return new_image

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None


# =========== EXAMPLE ==================

# ex_img = cv2.imread('yes/Y10.jpg')  # Load image
# ex_new_img = crop_brain_contour(ex_img, True)  # Process and plot

# ===================================== LOADING UP THE DATA =====================================

def load_data(dir_list, image_size):
    X, y = [], []
    image_width, image_height = image_size
    skipped = 0

    for directory in dir_list:
        for filename in listdir(directory):
            
            from os import path
            image = cv2.imread(path.join(directory, filename))
            cropped_image = crop_brain_contour(image, plot=False)
            
            if cropped_image is None:
                skipped += 1
                continue  # Skip this image if it couldn't be processed
            
            cropped_image = cv2.resize(cropped_image, (image_width, image_height))
            cropped_image = cropped_image / 255.0  # Normalize
            X.append(cropped_image)
            y.append([1] if directory[-3:] == 'yes' else [0])

    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    
    print(f'Number of examples: {len(X)}')
    print(f'Number of skipped images: {skipped}')
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    
    return X, y


# ===================================== LOADING UP THE  AUGUMENTED DATA =====================================


augmented_path = 'augmented data/'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + 'yes' 
augmented_no = augmented_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

# ===================================== PLOT THE SAMPLE IMAGES =====================================

def plot_sample_images(X, y, n=50):
    """
    Plots n sample images for both values of y (labels).
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """
    
    for label in [0,1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]
        
        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(20, 10))
        
        i = 1 # current plot        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])
            
            # remove ticks
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            
            i += 1
        
        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()

plot_sample_images(X, y)

# ===================================== SPLIT THE DATA =====================================

def split_data(X, y, test_size=0.2):
       
    """
    Splits data into training, development and test sets.
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    Returns:
        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
        y_train: A numpy array with shape = (#_train_examples, 1)
        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
        y_val: A numpy array with shape = (#_val_examples, 1)
        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
        y_test: A numpy array with shape = (#_test_examples, 1)
    """
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of development examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val (dev) shape: " + str(X_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

# ========= HELPER FUNCTIONS ===============

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"

#============ F1 SCORE =============

def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score

# ===================================== BUILDING THE NEURAL MODEL =====================================

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

# ========= DEFINE THE IMAGE SHAPE =========

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

model = build_model(IMG_SHAPE)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath="cnn-parameters-improvement-{epoch:02d}-{val_acc:.2f}"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(
    filepath="models/cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

tensorboard = TensorBoard(log_dir="./logs")

# ===================================== EVALUATE AND LOG RESULTS FOR THE MODEL =====================================

def get_last_iteration(csv_file):
    """Returns the last iteration number from the CSV file or 0 if the file doesn't exist or is empty."""
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:  # If only the header or empty file
            return 0
        last_line = lines[-1].split(",")[0].replace("Iteration ", "").strip()
        return int(last_line)

def evaluate_and_log_model(model, X_test, y_test, csv_file="accuracy_scores.csv"):
    """
    Evaluates the model and logs metrics into a CSV file with an auto-incremented iteration number.
    """
    start_time = time.time()
    
    # Simulating prediction for now. Replace with real model prediction.
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate compilation time
    compilation_time = time.time() - start_time
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the last iteration and increment it
    iteration = get_last_iteration(csv_file) + 1
    
    # Prepare the row to append
    row = [
        f"Iteration {iteration}",
        formatted_time,
        round(compilation_time, 2),
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(auc, 4),
        json.dumps(cm.tolist())
    ]
    
    # Append the metrics to the CSV
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write the header row
                writer.writerow([
                    "Iteration", "Compilation Time", "Execution Time (s)", "Accuracy", 
                    "Precision", "Recall", "F1 Score", "ROC AUC", "Confusion Matrix"
                ])
            writer.writerow(row)
        print(f"Metrics for Iteration {iteration} logged to {csv_file}.")
    except Exception as e:
        print(f"Error while writing to CSV: {e}")

# ===================================== CLASSIFICATION FOR THE IMAGE FUNCTION =====================================


def classify_and_move_images(folder_path, model, target_size, true_folder, false_folder):
    """
    Classifies all images in a folder and moves them to the corresponding true/false folder.
    
    Arguments:
        folder_path: str, path to the folder containing images to classify.
        model: Trained Keras model.
        target_size: tuple, (width, height) for resizing the images.
        true_folder: str, path to the folder for positive classification.
        false_folder: str, path to the folder for negative classification.
    """
    try:
        # Create target folders if they don't exist
        os.makedirs(true_folder, exist_ok=True)
        os.makedirs(false_folder, exist_ok=True)
        
        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            
            # Check if it's an image (basic check)
            if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
                print(f"Skipping non-image file: {filename}")
                continue

            # Load and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue

            processed_image = crop_brain_contour(image, plot=False)
            if processed_image is None:
                print(f"Could not process image: {filename}")
                continue
            
            processed_image = cv2.resize(processed_image, target_size)
            processed_image = processed_image / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)

            # Make prediction
            prediction = model.predict(processed_image)
            label = int(prediction > 0.5)

            # Determine the target folder and move the image
            target_folder = true_folder if label == 1 else false_folder
            shutil.move(image_path, os.path.join(target_folder, filename))
            print(f"Image '{filename}' classified as {'True' if label == 1 else 'False'} and moved to '{target_folder}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

# ===================================== TRAIN THE MODEL =====================================

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")
iteration = 1
evaluate_and_log_model(model, X_test, y_test, iteration)

# ===================================== EVALUATE THE MODEL(TEST ON AN IMPUT IMAGE)==============================
# folder_path = '/Users/macbook/Desktop/MASTER/ML/NeuralSense-master/TEST  folder'
# true_folder = '/Users/macbook/Desktop/MASTER/ML/NeuralSense-master/CLASSIFICATION result/true'
# false_folder = '/Users/macbook/Desktop/MASTER/ML/NeuralSense-master/CLASSIFICATION result/false'
# classify_and_move_images(folder_path, model, (IMG_WIDTH, IMG_HEIGHT), true_folder, false_folder)



