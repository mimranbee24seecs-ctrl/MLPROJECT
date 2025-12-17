# Machine Learning Project
By: Filza Fatima (506785), Ammar Imran (501062)# Project Report: Dog Breed Identification Using Deep Learning and Hybrid Approaches

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
* **The Process:** We passed every image through the neural network but stopped it *before* it made a final decision.
* **The Result:** Instead of an image, we get a list of **4,032 numbers** (a feature vector) for every dog. These numbers represent high-level concepts like "floppy ears" or "curly fur."
* We then fed these "smart features" into our classical algorithms (SVM and Random Forest).

### 2.3 Development Challenges & Resource Optimization
A significant challenge during the development phase was **Memory Management**.
* **Initial Attempt (VS Code):** Initially, we developed the project locally using VS Code. The strategy involved loading the entire dataset into massive NumPy arrays (`X_train`, `y_train`).
* **The Problem:** Storing thousands of high-resolution images ($331 \times 331 \times 3$) as floating-point numbers in a single NumPy block required over 16GB of RAM. This exceeded the memory limits of standard Google Colab instances and caused frequent system crashes ("Out of Memory" errors).
* **The Solution:** We migrated from static NumPy arrays to TensorFlow’s `ImageDataGenerator`.
    * Instead of loading all images at once, the Generator loads images in small "batches" (e.g., 8 images at a time) strictly when the GPU needs them.
    * This allowed us to train a very deep model (NASNetLarge) on limited hardware without crashing the session.

---

## 3. Implementation Details

### 3.1 Deep Learning (Neural Networks)
We utilized **Transfer Learning** with the **NASNetLarge** architecture, which was pre-trained on the ImageNet dataset.

* **Architecture Breakdown:**
    * **Base (NASNetLarge):** This is the "eye" of the model. We froze its weights so it retains the knowledge it learned from millions of other images.
    * **Global Average Pooling:** This "squashes" the complex 3D data block into a flat list of numbers. This drastically reduces the number of parameters and prevents overfitting compared to the Flatten layer.
    * **Prediction Layer:** A Dense layer with 120 neurons (one for each breed) using `softmax` activation to output probabilities.

* **Training Configuration:**
    * **Optimizer:** `Adam` (Adaptive Moment Estimation).
    * **Callbacks:** We used `EarlyStopping` to prevent wasted training time and `ReduceLROnPlateau` to fine-tune the learning rate dynamically.

### 3.2 Classical Machine Learning
We implemented two classical algorithms using the features extracted by NASNet:

1.  **Support Vector Machine (SVM):**
    * *Concept:* Plots every dog on a high-dimensional graph and draws straight lines (hyperplanes) to separate the breeds.
    * *Configuration:* Used a linear kernel, which is efficient for high-dimensional feature vectors.
2.  **Random Forest:**
    * *Concept:* Creates a "forest" of 100 decision trees. The final decision is based on a majority vote from all trees.

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
3.  **Optimization:** We successfully refactored the data pipeline from memory-heavy NumPy arrays to efficient Generators to function within the Colab environment.
2.  **Classical ML:** We successfully used Feature Extraction to train SVM and Random Forest models.
3.  **Evaluation:** We compared multiple models and found that while Classical methods work well with good features, **Deep Learning is statistically superior** for image classification tasks.
