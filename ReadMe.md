🦷 Teeth Disease Classification using CNN

Overview

Welcome to the Teeth Disease Classification project! 🌟 Imagine a future where dental issues are diagnosed instantly, saving countless smiles. This project brings us a step closer to that reality by leveraging the power of deep learning to classify teeth images into one of seven disease categories with an impressive accuracy of 95%! 🎉


Table of Contents

Introduction

Dataset

Preprocessing

Model Architecture

Training

Results

How to Use

Conclusion

Future Work

Introduction

Dental health is more than just a beautiful smile; it’s a key to overall wellness. However, diagnosing dental diseases can be challenging and time-consuming. This project, powered by a Convolutional Neural Network (CNN), aims to change that by automating the classification of teeth images into seven distinct disease categories. With this technology, we’re one step closer to revolutionizing dental care. 🦷💡

Dataset

The heart of this project lies in a carefully curated dataset of teeth images, each labeled with one of seven possible diseases. These images have been meticulously resized to 128x128 pixels and rescaled, ensuring they’re ready to be fed into our CNN model.

Preprocessing

To bring out the best in our model, we’ve gone through a thoughtful preprocessing pipeline:

Image Resizing: Each image is resized to a neat 128x128 pixels, giving our model the consistency it craves.
Rescaling: We’ve rescaled pixel values to [0, 1], setting the stage for optimal learning.
Image Augmentation: By applying transformations like rotation, zoom, and flipping, we’ve armed our model with the resilience to handle real-world variations. 🛠️
Model Architecture
Our CNN is built with layers of brilliance:

Convolutional Layers: These layers are like detectives, hunting down features in the images.
Pooling Layers: They trim down the noise, making the model sharper and quicker.
Fully Connected Layers: The final decision-makers, classifying the images based on the features detected.
Activation Functions: ReLU adds the energy, and Softmax brings clarity to the output.
With Adam as our optimizer and categorical cross-entropy as our guide, we’ve crafted a model that’s both powerful and precise.

Training

Training this model was a journey of discovery and fine-tuning:

Batch Size: 32 — just the right amount for balanced learning.
Epochs: 50 — the sweet spot for mastering the dataset.
Validation Split: A wise 20% reserved for validation, keeping us honest throughout the process.
Early Stopping: A safeguard against overfitting, ensuring our model doesn’t just memorize but truly learns.
Results
And the results? A stunning 95% accuracy on the test dataset! 🏆 This model is not just reliable—it’s a game-changer in the world of dental diagnostics.

How to Use

Prerequisites

Before you dive in, make sure you have these tools ready:

Python 3.x

TensorFlow/Keras

NumPy

OpenCV

Matplotlib

Steps

Clone this repository to your local machine.

git clone https://github.com/salmamuhammede/Teeth-disease-classification-CNN-Cellula-task-

Step into the project directory.

cd teeth-disease-classification

Install the necessary dependencies.

pip install -r requirements.txt

Prepare the dataset and place it where it belongs.
Run the training script and watch the magic happen.

python train.py
Test the model on a new image—who knows what it will discover?

python test.py --image_path /path/to/your/image.jpg

Conclusion

In this project, we’ve witnessed the incredible potential of CNNs in the field of dental health. The 95% accuracy achieved is not just a number—it’s a testament to the power of AI in transforming how we diagnose and treat dental diseases. With this model, the future of dental care looks brighter than ever. 🌟

Future Work

The journey doesn’t end here! Here’s what’s next:

Expand the Dataset: More images mean a smarter, more robust model. 📈
Fine-Tuning: There’s always room for improvement—experimenting with different architectures could push the accuracy even higher.
Deployment: Imagine this model in the hands of dentists worldwide! Developing a web or mobile app could make that a reality. 🚀