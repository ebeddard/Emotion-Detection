import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D

# This will only work if you are in the same directory as the faces folder
# Get the path to the image directory and the .anonr file
path_to_images = os.getcwd() + "/faces"
anonr_file = os.path.join(path_to_images, ".anonr")

# Load the .anonr file and get the list of all usernames
with open(anonr_file, "r") as f:
    usernames = f.read().splitlines()

# Create an empty list to hold the image data and labels
data = []

# Set the new dimensions for resizing the image
width = 64
height = 64

# Loop through each username
for username in usernames:
    # Construct the path to the username folder
    user_folder = os.path.join(path_to_images, username)

    # Process each image file in the username folder
    for root, dirs, files in os.walk(user_folder):
        for file in files:
            # Confirm the file has the .pgm extension
            if file.endswith(".pgm"):
                # Construct the full path to the image file
                image_file = os.path.join(root, file)

                # Load the image
                img = cv2.imread(image_file)

                # Convert to gray-scale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize the image
                resized = cv2.resize(gray, (width, height))

                # Normalize the image to have values between 0 and 1
                normalized = resized.astype(np.float32) / 255.0

                # Set the emotion using the label in the image name
                filename_parts = file.split("_")
                emotion_label = filename_parts[2]
                if(emotion_label == "angry"):
                    emotion = 0
                elif(emotion_label == "happy"):
                    emotion = 1
                elif(emotion_label == "neutral"):
                    emotion = 2
                elif(emotion_label == "sad"):
                    emotion = 3
                else:
                    emotion = -1
                   
                # Set face label using the image file name components
                face_label = filename_parts[3] 
                if(face_label == "open"):
                    face = 0
                elif(face_label == "sunglasses"):
                    face = 1
                else:
                    face = -1

                # Append the image data and labels to the list
                data.append((normalized, emotion, username, face))

# Create a data frame from the list of image data and labels
df = pd.DataFrame(data, columns=["Image", "Emotion", "Username", "Face"])

# Get the rows from df that have open faces (no sunglasses) 
df_open_face = df[df["Face"] == 0]

# Print the number of samples in both data frames
print(f"Total samples: {df.shape[0]}")
print(f"No sunglasses samples: {df_open_face.shape[0]}")

# Use train_test_split to split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Get the images and emotion labels for training
x_train = np.array(train_df["Image"].tolist())
y_train = np.array(train_df["Emotion"].tolist())

# Get the images and emotion labels for testing
x_test = np.array(test_df["Image"].tolist())
y_test = np.array(test_df["Emotion"].tolist())

# Create the layers and add them to the Convolutional Neural Network
cnn_layer1 = Input((64,64,1))
cnn_layer2 = Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_layer3 = MaxPool2D(pool_size=(2, 2))
cnn_layer4 = Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_layer5 = MaxPool2D(pool_size=(2, 2))
cnn_layer6 = Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_layer7 = MaxPool2D(pool_size=(2, 2))
cnn_layer8 = Flatten()
cnn_layer9 = Dense(4,activation='softmax')
cnn_model = Sequential()
cnn_model.add(cnn_layer1)
cnn_model.add(cnn_layer2)
cnn_model.add(cnn_layer3)
cnn_model.add(cnn_layer4)
cnn_model.add(cnn_layer5)
cnn_model.add(cnn_layer6)
cnn_model.add(cnn_layer7)
cnn_model.add(cnn_layer8)
cnn_model.add(cnn_layer9)

# Output model details
cnn_model.summary()

# Train!
cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_epochs = 60
history = cnn_model.fit(x_train.reshape(-1, 64, 64 ,1), y_train, epochs=n_epochs,
                        validation_data=(x_test.reshape(-1, 64, 64 ,1), y_test))


# Now do the same thing as above but for the data without sunglasses

# Use train_test_split to split data
train_df_open_face, test_df_open_face = train_test_split(df, test_size=0.2, random_state=0)

# Get the images and emotion labels for training
x_train_open_face = np.array(train_df_open_face["Image"].tolist())
y_train_open_face = np.array(train_df_open_face["Emotion"].tolist())

# Get the images and emotion labels for testing
x_test_open_face = np.array(test_df_open_face["Image"].tolist())
y_test_open_face = np.array(test_df_open_face["Emotion"].tolist())

# Create the layers and add them to the Convolutional Neural Network
cnn_open_layer1 = Input((64,64,1))
cnn_open_layer2 = Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_open_layer3 = MaxPool2D(pool_size=(2, 2))
cnn_open_layer4 = Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_open_layer5 = MaxPool2D(pool_size=(2, 2))
cnn_open_layer6 = Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=(64,64,1))
cnn_open_layer7 = MaxPool2D(pool_size=(2, 2))
cnn_open_layer8 = Flatten()
cnn_open_layer9 = Dense(4,activation='softmax')
cnn_open_model = Sequential()
cnn_open_model.add(cnn_open_layer1)
cnn_open_model.add(cnn_open_layer2)
cnn_open_model.add(cnn_open_layer3)
cnn_open_model.add(cnn_open_layer4)
cnn_open_model.add(cnn_open_layer5)
cnn_open_model.add(cnn_open_layer6)
cnn_open_model.add(cnn_open_layer7)
cnn_open_model.add(cnn_open_layer8)
cnn_open_model.add(cnn_open_layer9)

# Output model details
cnn_open_model.summary()

# Train!
cnn_open_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_open = cnn_open_model.fit(x_train_open_face.reshape(-1, 64, 64 ,1), y_train_open_face, epochs=n_epochs,
                        validation_data=(x_test_open_face.reshape(-1, 64, 64 ,1), y_test_open_face))

# Print the results from using entire dataset and only using images without sunglasses
print(f"\nAccuracy on the final epoch of training on all data was {100*history.history['accuracy'][-1]:0.2f}%")

print(f"\nAccuracy on the final epoch of training on only open faces was {100*history_open.history['accuracy'][-1]:0.2f}%")