# Cell Anomaly Detection using Autoencoders

This repository provides an implementation of an anomaly detection system for cell images using autoencoders. The project draws inspiration from the paper "Robust Anomaly Detection in Images using Adversarial Autoencoders" by Laura Beggel, Michael Pfeiffer, and Bernd Bischl.

## Project Structure

The project focuses on detecting anomalies in images using autoencoder neural networks. An autoencoder learns to reconstruct normal images and can classify images as anomalies when the reconstruction error exceeds a certain threshold. The code in this repository implements an autoencoder-based anomaly detection method using TensorFlow.

### Overview

![Overview Diagram](/docs/assets/overview.png)

The project addresses a fundamental challenge in anomaly detection using autoencoders, particularly when the training set contains outliers. Continued training of autoencoders tends to reduce the reconstruction error of outliers, thereby degrading anomaly detection performance. To mitigate this issue, an adversarial autoencoder architecture is employed, which imposes a prior distribution on the latent representation, typically placing anomalies into low likelihood regions. By utilizing the likelihood model, potential anomalies can be identified and rejected during training, resulting in a more robust anomaly detector.

### Architecture

![Architecture Diagram](/docs/assets/architecture.png)

The architecture employed in this project leverages a VGG16-based model, modified for the task of encoding and decoding images for anomaly detection. Here's a breakdown of how the architecture is structured:

- **Encoder:** The encoder part of the architecture is based on the VGG16 model, with the fully connected layers removed. The final layer of this modified VGG16 encoder outputs a 7x7x7 encoded image vector. This condensed representation captures the essential features of the input images.

- **Kernel Density Estimation (KDE):** In the middle of the architecture, Kernel Density Estimation (KDE) is used to calculate the likelihood of an image belonging to the 'good' class. KDE is applied to the training data to provide an estimate of where the input image vector space lies within the latent space. This estimation helps in determining the 'normal' density regions.

- **Decoder:** The decoder mirrors the architecture of the encoder but in reverse. It takes the encoded vector and reconstructs the image. The quality of reconstruction is crucial for detecting anomalies.

- **Anomaly Detection:** Anomalies are determined based on two criteria:
  1. **KDE Value:** If the KDE of an image's latent representation is below a certain threshold, the image is considered an anomaly. This threshold is set based on the density distribution of the training images. Images with latent representations that lie far from the high-density regions (where most training images lie) are flagged as anomalies.
  2. **Reconstruction Error:** Additionally, if the reconstruction error (the difference between the original image and its reconstructed image from the decoder) exceeds a predefined threshold, the image is also classified as an anomaly.

This dual-criterion approach helps in robustly identifying images that do not conform to the learned distribution of 'normal' images, either through significant deviations in their latent space positioning or through poor reconstruction quality.





## Repository Structure

- **`docs`:** Contains documentation files related to the project.

- **`notebook`:** Holds Jupyter notebook files.

- **`pretrained_models`:** Contains pretrained models saved in various formats.

- **`scripts`:** Holds Python scripts used in the project.

- **`src`:** Contains the source code files of the project.
    - [main.py](/src/main.py): Python script orchestrating the workflow by calling functions from other modules.
    - [data.py](/src/data.py): Module handling data loading.
    - [train.py](/src/train.py): Module dealing with model building and training.
    - [evaluate.py](/src/evaluate.py): Module including functions for calculating the density and reconstruction errors.
    - [visualize.py](/src/visualize.py): Module containing the function for plotting the training and validation loss.


## Prerequisites

![Python](https://img.shields.io/badge/Python-3.9.2-blue)
![Pillow](https://img.shields.io/badge/Pillow-8.3.2-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-yellow)
![numpy](https://img.shields.io/badge/numpy-1.21.2-blue)
![tensorflow](https://img.shields.io/badge/tensorflow-2.6.0-green)

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Use

**Step 1: Download Dataset** [Malaria Cell Images Dataset](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip)

After downloading, unzip the dataset and place it in the appropriate directory.

**Step 2: Data Preparation**

Split the downloaded dataset into training and testing sets using the `split.py` script.

```bash
python scripts/split.py
```

**Step 3: Training the Model**

Train the autoencoder model using the training data.

```bash
python src/train.py
```

**Step 4: Evaluating the Model**

Evaluate the model to calculate density and reconstruction errors.

```bash
python src/evaluate.py
```

**Step 5: Visualizing the Results**

Visualize the training and validation loss.

```bash
python src/visualize.py
```

**Step 6: Running the Full Pipeline**

You can run the entire pipeline using the `main.py` script.

```bash
python src/main.py
```

## Pretrained Models

Pretrained models are provided in the `pretrained_models` directory. You can load and use these models directly without training:

* `cell_anomaly_detection.h5`
* `cell_anomaly_detection_encoder.h5`
* `my_model.keras`

## Reference

This project is based on the paper:
- Laura Beggel, Michael Pfeiffer, Bernd Bischl. "Robust Anomaly Detection in Images using Adversarial Autoencoders" [arXiv:1901.06355 [cs.LG]](https://arxiv.org/abs/1901.06355)

The detailed methodology and experimental results are available in the paper included in the `docs` directory.

## Contributing

Contributions are welcome! 

- **Bug Fixes:** If you find any bugs or issues, feel free to create an issue or submit a pull request.
- **Feature Enhancements:** If you have ideas for new features or improvements, don't hesitate to share them.

## License

This project is licensed under the [MIT License](LICENSE). 
