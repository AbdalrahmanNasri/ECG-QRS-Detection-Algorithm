import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# Load the model
model = load_model(r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\Project1\model\model_CV.h5')

# Load the a test image
img_path = r'F:\7.jpeg'
img = cv2.imread(img_path)  # Use OpenCV to read the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
img = cv2.resize(img, (224, 224))  # Resize the image

# Reshape the image to add a batch dimension
img = np.expand_dims(img, axis=0)

# Normalize the image if your model was trained on normalized images
img = img / 255.0

# Make a prediction
predictions = model.predict(img)

predictions = predictions.reshape(-1)

# Determine the threshold for decision boundary (0.5)
decision_boundary = (predictions >= 0.5).astype(int)
print(decision_boundary)
