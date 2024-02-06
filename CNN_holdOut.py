import keras
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, AveragePooling2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold

# image deminsion
H = 224
W = 224

# loading data 
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(H, W))
        img_array = img_to_array(img)
        if img is not None:
            images.append(img_array)
            labels.append(label)
    return images, labels

images_class_1, labels_class_1 = load_images_from_folder(r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\Project1\dataSet\not_QRS', 0)
# print(f"len1: {len(images_class_1)}")
images_class_2, labels_class_2 = load_images_from_folder(r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\Project1\dataSet\QRS', 1)
# print(f"len2: {len(images_class_2)}")

# Combining the data and labels
data = np.vstack([images_class_1, images_class_2])
labels = np.hstack([labels_class_1, labels_class_2])

# HouldOut method
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# CNN
model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(5,5), input_shape=(H,W,3),activation='relu'))

model.add(AveragePooling2D(pool_size=(8,8)))
model.add(Conv2D(filters=8,kernel_size=(5,5), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=4,kernel_size=(4,4),activation='relu'))
    
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))  # fully aconnected layer
model.add(Dense(units=1, activation='sigmoid'))  # sigmoid function for binary classification

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[keras.metrics.BinaryAccuracy()])
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# training the model
model.fit(X_train, y_train, epochs= 15, batch_size= 32)

# print("###########################################")

# model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


 
# saving the model
model.save(r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\Project\model_holdout.h5')
 