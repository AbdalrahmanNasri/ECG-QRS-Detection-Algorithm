from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import keras
import os
from sklearn import metrics
import matplotlib.pyplot as plt

# load the module
model = load_model(r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\Project1\model\model_CV.h5')


H = 224
W = 224

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

# Combine the data and labels
data = np.vstack([images_class_1, images_class_2])
labels = np.hstack([labels_class_1, labels_class_2])

# spliting data using cross_validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# K-Fold Cross-Validation
fold = 1
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

y_pred_probs = model.predict(X_test)  

# Set a threshold 
threshold = 0.5

# Convert probabilities to binary labels using the threshold
y_pred_binary = np.where(y_pred_probs > threshold, 1, 0)

# Calcuation of the roc Score
auc_roc = roc_auc_score(y_test, y_pred_probs)
print(f"AUC-ROC: {auc_roc}")

# ploting Roc Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# MSE calculation
def calculate_mse(gt, prediction):
    # Calculate the differences
    differences = (np.array(prediction) - np.array(gt))**2 /(len(gt)) 

    # Square the differences and compute the mean
    mean_squared_error = np.mean(np.square(differences))

    # Return the square root of the mean squared error
    return mean_squared_error

 
ground_truth = labels      
predicted = y_pred_probs  

mse_value = calculate_mse(ground_truth, predicted)
print("MSE Value:", mse_value)
print("RMS Value:", np.sqrt(mse_value))


# confusion matrix
confusion_metrix = metrics.confusion_matrix(y_test, y_pred_binary)


def cnMetrix(confusion_matrix):
    TN = confusion_metrix[0, 0]
    TP = confusion_metrix[1, 1]
    FP = confusion_metrix[0, 1]
    FN = confusion_metrix[1, 0]
    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # recall
    recall = TP / (TP + FN)
    # Specifity
    specifity = TN / (TN + FP)
    # precision
    precision = TP / (TP + FP)
    # F1 Score
    F_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"specifity: {specifity}")
    print(f"F1 Score: {F_score}")

cnMetrix(confusion_metrix)

# ploting confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_metrix, display_labels = [False, True])

cm_display.plot()
plt.show()

