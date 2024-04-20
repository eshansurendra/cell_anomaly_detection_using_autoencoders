# Image Anomaly Detection using Autoencoders

This repository contains code for image anomaly detection using autoencoders.

## Description
The project focuses on detecting anomalies in images using autoencoder neural networks. An autoencoder learns to reconstruct normal images and can classify images as anomalies when the reconstruction error exceeds a certain threshold. The code in this repository implements an autoencoder-based anomaly detection method using TensorFlow.

## Usage
1. **Preparation**: Ensure you have all the necessary packages installed, including TensorFlow.

2. **Dataset Download**: Download the dataset from the following link:
   [Malaria Cell Images Dataset](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip)

3. **Dataset Splitting**: After downloading the dataset, split it into train and test sets using the provided Python script `split.py`. Here's how to use the script:
   
   ```bash
   python split.py
   ```

   The script will divide the dataset into train and test directories with the specified ratio.

4. **Model Loading**: Load the pre-trained autoencoder model and encoder model from the provided files (`cell_anomaly_detection.h5` and `cell_anomaly_detection_encoder.h5`).

5. **Model Compilation**: Compile the loaded models manually if no training configuration is found.

6. **Data Preparation**: Prepare your image data and set up data generators for training.

7. **Model Evaluation**: Use the provided functions to evaluate the model's performance on used dataset.

8. **Anomaly Detection**: Use the `check_anomaly` function to detect anomalies in your images.

## Implementation Details
- **Kernel Density Estimation (KDE)**: Kernel Density Estimation is used to calculate the density of encoded image vectors.
- **Model Evaluation**: The model is evaluated based on both KDE and reconstruction error thresholds.
- **Anomaly Detection**: Anomaly detection is performed by comparing the density of encoded image vectors and reconstruction errors with predefined thresholds.

## Reference
This project is based on the paper:
- Laura Beggel, Michael Pfeiffer, Bernd Bischl. "Robust Anomaly Detection in Images using Adversarial Autoencoders" [arXiv:1901.06355 [cs.LG]](https://arxiv.org/abs/1901.06355)

## License
The code and resources in this repository are provided as-is and may have specific usage terms. Please refer to the provided paper for more information.
