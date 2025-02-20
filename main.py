import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import seaborn as sns

#Trying to run using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is GPU available:", tf.test.is_gpu_available())
print("GPU Device Name:", tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU:", physical_devices[0])
else:
    print("No GPU detected. Using CPU.")


# Define paths
data_dir = "dataset"
categories = ["benign", "malignant", "normal"]

# Load and preprocess data with masks and texture features
def load_data_with_masks_and_features(data_dir, categories):
    print("Loading images, masks, and extracting texture features...")
    images = []
    labels = []
    glcm_features = []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            if file.endswith(".png") and "mask" not in file:
                # Load image
                img_path = os.path.join(category_path, file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))  # Adjusted for InceptionResNetV2
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize

                # Load mask
                mask_path = img_path.replace(".png", "_mask.png")
                if os.path.exists(mask_path):
                    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(299, 299), color_mode="grayscale")
                    mask_array = tf.keras.preprocessing.image.img_to_array(mask) / 255.0
                    img_array = img_array * mask_array  # Apply mask

                # Extract GLCM features
                gray_img = (img_array[..., 0] * 255).astype(np.uint8)  # Convert to grayscale
                glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]

                images.append(img_array)
                labels.append(label)
                glcm_features.append([contrast, energy, homogeneity, correlation])

    print("Data loading and feature extraction complete.")
    return np.array(images), np.array(labels), np.array(glcm_features)

# Load dataset
images, labels, glcm_features = load_data_with_masks_and_features(data_dir, categories)

# Train-test split
X_train_img, X_test_img, X_train_glcm, X_test_glcm, y_train, y_test = train_test_split(
    images, glcm_features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert labels to categorical
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=len(categories))
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(categories))

# Data augmentation
print("Applying data augmentation...")
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_gen = data_gen.flow(X_train_img, y_train_categorical, batch_size=32)

# Build InceptionResNetV2 model
print("Building and fine-tuning InceptionResNetV2 model...")
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# GLCM Feature Input
glcm_input = Input(shape=(4,))
glcm_dense = Dense(128, activation='relu')(glcm_input)

# Concatenate Image and GLCM Features
combined = Concatenate()([x, glcm_dense])
combined = Dense(256, activation='relu')(combined)
combined = Dropout(0.5)(combined)
output = Dense(len(categories), activation='softmax')(combined)

# Final Model
model = Model(inputs=[base_model.input, glcm_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(
    [X_train_img, X_train_glcm], y_train_categorical,
    validation_data=([X_test_img, X_test_glcm], y_test_categorical),
    epochs=30, batch_size=16
)
print("Training complete.")

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate([X_test_img, X_test_glcm], y_test_categorical)
print(f"Test Accuracy: {accuracy * 100:.2f}%") #98.08%

# Generate Predictions
y_pred = model.predict([X_test_img, X_test_glcm])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_categorical, axis=1)

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Generating classification report...")
report = classification_report(y_true, y_pred_classes, target_names=categories, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report)

# Accuracy Graph
print("Plotting training and validation accuracy...")
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization: Bar Chart for Precision, Recall, and F1-Score
print("Generating bar chart for precision, recall, and F1-score...")
df_metrics = df_report.iloc[:-3, :3]  # Exclude support and avg/total rows
df_metrics.plot(kind='bar', figsize=(10, 6))
plt.title("Precision, Recall, and F1-Score by Class")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# Display Key Results in a Table
print("Displaying key results in a table...")
results_summary = {
    'Metric': ['Accuracy', 'Precision (Macro Avg)', 'Recall (Macro Avg)', 'F1-Score (Macro Avg)'],
    'Score': [
        accuracy_score(y_true, y_pred_classes),
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score']
    ]
}

# Save the trained model
model.save("breast_tumor_classifier.h5")
print("Model saved as 'breast_tumor_classifier.h5'")

loaded_model = tf.keras.models.load_model("breast_tumor_classifier.h5")
print("Model loaded successfully.")

df_summary = pd.DataFrame(results_summary)
print(df_summary)
