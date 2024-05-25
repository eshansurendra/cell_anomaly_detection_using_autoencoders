# Cell Anomaly Detection using Autoencoders

This repository provides an implementation of an anomaly detection system for cell images using autoencoders. The project draws inspiration from the paper "Robust Anomaly Detection in Images using Adversarial Autoencoders" by Laura Beggel, Michael Pfeiffer, and Bernd Bischl.

## Project Structure

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
