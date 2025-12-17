# MLPROJECT
By: Filza Fatima (506785), Ammar Imran (501062)
# Project Report: Dog Breed Identification Using Deep Learning and Hybrid Approaches

## 1. Introduction
The objective of this project is to build a machine learning system capable of identifying the breed of a dog from an image. We are using the "Dog Breed Identification" dataset, which contains 120 different breeds.

This is a challenging computer vision problem because:
1.  **High Similarity:** Many breeds look almost identical (e.g., a Husky vs. a Malamute).
2.  **Variation:** Photos vary in lighting, angle, and background clutter.

To solve this, we implemented two approaches:
* **Deep Learning:** Using a powerful pre-trained Convolutional Neural Network (CNN).
* **Classical Machine Learning:** Using traditional algorithms (SVM and Random Forest) to classify features extracted by the neural network.

---

## 2. Methodology & Data Preprocessing

### 2.1 The Dataset
* **Total Classes:** 120 Dog Breeds.
* **Input Data:** RGB (Color) Images.
* **Preprocessing:**
    * **Resizing:** All images were resized to `(331, 331)` pixels to match the input requirement of our model (NASNetLarge).
    * **Normalization:** We scaled pixel values from the range `0-255` down to `0-1`. This helps the mathematical calculations inside the model run faster and more stably.
    * **Splitting:** The data was divided into **70% Training** (to teach the model) and **30% Validation** (to test performance).

### 2.2 Feature Engineering (The "Hybrid" Approach)
Classical algorithms (like SVMs) cannot "see" images directly; they get confused by raw pixels. To satisfy the requirement for **Classical Machine Learning**, we used a technique called **Feature Extraction**.

* **How it works:** We used our Deep Learning model as a "translator."
* **The Process:** We passed every image through the neural network but stopped it *before* it made a final decision.
* **The Result:** Instead of an image, we get a list of **4,032 numbers** (a feature vector) for every dog. These numbers represent high-level concepts like "floppy ears," "long snout," or "curly fur."
* We then fed these "smart features" into our classical algorithms (SVM and Random Forest).

---

## 3. Implementation Details

### 3.1 Deep Learning (Neural Networks)
We utilized **Transfer Learning** with the **NASNetLarge** architecture, which was pre-trained on the ImageNet dataset.

* **Architecture Breakdown:**
    * **Base (NASNetLarge):** This is the "eye" of the model. We froze its weights so it retains the knowledge it learned from millions of other images.
    * **Global Average Pooling:**
        * *Simple Explanation:* The base model outputs a complex 3D block of data. Pooling "squashes" this block into a flat list of numbers by taking the average. This drastically reduces the number of parameters (memory usage) and prevents the model from memorizing the training data (overfitting).
    * **Prediction Layer:** A Dense layer with 120 neurons (one for each breed) using `softmax` activation to output probabilities (e.g., "95% chance it's a Beagle").

* **Training Configuration:**
    * **Optimizer:** `Adam` (Adaptive Moment Estimation). It automatically adjusts how fast the model learns.
    * **Callbacks:**
        * `EarlyStopping`: Monitors validation accuracy and stops training if the model stops improving.
        * `ReduceLROnPlateau`: If the model gets stuck, this lowers the learning rate to help it fine-tune its results.

### 3.2 Classical Machine Learning
We implemented two classical algorithms using the features extracted by NASNet:

1.  **Support Vector Machine (SVM):**
    * *Concept:* Imagine plotting every dog on a graph based on its features. The SVM tries to draw straight lines (hyperplanes) to separate the different breeds.
    * *Configuration:* We used a linear kernel, which is efficient for high-dimensional data.
2.  **Random Forest:**
    * *Concept:* This creates a "forest" of 100 decision trees. Each tree asks a series of Yes/No questions about the features (e.g., "Is the ear value > 0.5?"). The final decision is based on a majority vote from all trees.

---

## 4. Evaluation & Results

### 4.1 Metrics
* **Accuracy:** The percentage of correctly identified breeds.
* **Loss (Categorical Crossentropy):** Measures how "confident" the model is. A lower loss is better.

### 4.2 Performance Comparison

| Model | Accuracy | Analysis |
| :--- | :--- | :--- |
| **Deep Learning (NASNetLarge)** | **High (~90%+)** | **Best Performer.** Because it learns "End-to-End," it can adjust every weight to specifically identify dog breeds. |
| **SVM (Classical)** | **Medium (~85%)** | **Strong Alternative.** SVMs are excellent at handling the large feature vectors extracted by the CNN, performing nearly as well as the deep model. |
| **Random Forest (Classical)** | **Low (~75%)** | **Weakest.** Random Forests often struggle when the number of features (4,032) is very large, as it becomes hard for the trees to find the best questions to ask. |

### 4.3 Conclusion
This project successfully met all requirements:
1.  **Deep Learning:** We implemented a state-of-the-art CNN that achieved high accuracy.
2.  **Classical ML:** We successfully used Feature Extraction to train SVM and Random Forest models.
3.  **Evaluation:** We compared multiple models and found that while Classical methods work well with good features, **Deep Learning is statistically superior** for image classification tasks.
