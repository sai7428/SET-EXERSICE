# Lung Segmentation in CT Scans using U-Net

This project demonstrates the segmentation of lungs from 2D CT scan images using a U-Net architecture. The model is built with TensorFlow and Keras. The entire workflow, from data loading and preprocessing to model training and evaluation, is contained within the `Lungs in CT Data FINAL.ipynb` notebook.

![Sample Prediction](./predicted_mask_visualization.png)
*Example of a CT scan, its ground truth mask, and the model's predicted mask.*

---

## 📖 Overview

The goal of this project is to automatically identify and outline the lung regions in chest CT scans. This is a common task in medical image analysis that can aid in diagnostics and treatment planning. The project uses a U-Net, a convolutional neural network designed for biomedical image segmentation.

The notebook covers the following key steps:
1.  **Data Acquisition**: Downloads the "Finding Lungs in CT Data" dataset from Kaggle.
2.  **Preprocessing**: Resizes, normalizes, and splits the images and masks into training, validation, and test sets.
3.  **Data Augmentation**: Applies random flips and rotations to the training data to improve model generalization.
4.  **Model Definition**: Implements a U-Net model from scratch using TensorFlow/Keras.
5.  **Custom Loss & Metrics**: Utilizes a combined Binary Cross-Entropy and Dice Loss (`bce_dice_loss`) for training, with Dice Coefficient and Intersection over Union (IoU) as performance metrics.
6.  **Training**: Trains the U-Net model for 25 epochs.
7.  **Evaluation & Visualization**: Evaluates the model's performance on the test set and visualizes the prediction results against the ground truth.

---

## 📊 Dataset

The project uses the **Finding Lungs in CT Data** dataset, publicly available on Kaggle. The dataset provides 2D CT scan images and their corresponding lung segmentation masks.

-   **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data)

The notebook automatically downloads and unzips the required `2d_images.zip` and `2d_masks.zip` files.

---

## 🛠️ Model Architecture

The segmentation is performed by a **U-Net**, which consists of an encoder (down-sampling path) to capture context and a decoder (up-sampling path) to enable precise localization. Skip connections between the encoder and decoder paths help combine high-level and low-level feature maps, leading to better segmentation performance.

-   **Loss Function**: `bce_dice_loss` (a combination of Binary Cross-Entropy and Dice Loss). This hybrid loss function is effective for handling class imbalance common in medical imaging.
-   **Evaluation Metrics**:
    -   `dice_coef` (Dice Coefficient)
    -   `iou_coef` (Intersection over Union / Jaccard Index)

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.x
-   A Kaggle account and API token (`kaggle.json`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [your-repository-directory]
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    **requirements.txt:**
    ```
    tensorflow
    kaggle
    opencv-python-headless
    scikit-learn
    matplotlib
    ```

3.  **Kaggle API Setup:**
    -   Download your `kaggle.json` API token from your Kaggle account page (`kaggle.com -> Your Profile -> Account -> Create New API Token`).
    -   The notebook is set up to upload this file directly in a Google Colab environment. If running locally, you'll need to place the `kaggle.json` file in the `~/.kaggle/` directory.

### Usage

1.  Open and run the `Lungs in CT Data FINAL.ipynb` notebook in a compatible environment (like Jupyter Notebook or Google Colab).
2.  When prompted, upload your `kaggle.json` file.
3.  The notebook will handle the rest, from downloading the data to training the model and showing the results.

---

## 📈 Results

The model was trained for 25 epochs and achieved the following performance on the test set:

-   **Test Loss**: 0.0679
-   **Test Dice Coefficient**: 0.9650
-   **Test IoU Score**: 0.9317


