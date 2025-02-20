# Breast Tumor Detection Using Inception-ResNet-v2

Welcome to the repository for my Breast Tumor Detection project, an advanced machine learning solution designed to enhance the accuracy and efficiency of diagnosing breast tumors using the powerful Inception-ResNet-v2 architecture.

## Project Overview

This project utilizes the Inception-ResNet-v2 model, tailored for high-precision image classification, to automatically detect breast tumors in ultrasound images. Achieving an impressive 98% accuracy, this model demonstrates significant potential in assisting medical professionals with early and accurate tumor detection.

## Dataset

The project uses the Breast Ultrasound Images Dataset (BUSI), available on Kaggle. This dataset includes labeled images categorized into normal, benign, and malignant classes, which are crucial for training our model. You can access the dataset [here](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset).

## Features

- **High Accuracy**: The model achieves 98% accuracy, providing reliable support for medical diagnosis.
- **Advanced Deep Learning Model**: Inception-ResNet-v2 is utilized for its robust performance in image recognition tasks.
- **User-Friendly Application**: Includes a web interface for easy interaction by healthcare professionals.

## Technologies Used

- **Python**: Primary programming language.
- **TensorFlow**: Used to build and train the Inception-ResNet-v2 model.
- **OpenCV**: Employed for image processing tasks.
- **Streamlit**: For deploying the model as a web application.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- Streamlit

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-tumor-detection.git
   cd breast-tumor-detection

## Model Training
The model was trained using the BUSI dataset. Preprocessing steps included resizing images to 299x299 pixels and normalizing them to match the input requirements of Inception-ResNet-v2. For detailed training procedures and model configurations, refer to the training_notebook.ipynb included in this repository.

## Contributing
Contributions are what make the open-source community such a fantastic place to learn, inspire, and create. Any contributions you make are greatly appreciated.

## Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
