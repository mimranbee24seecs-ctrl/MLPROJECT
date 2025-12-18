# Project Report: Dog Breed Identification Using Deep Learning and Hybrid Approaches

**By:** Filza Fatima (506785), Ammar Imran (501062)

---

## 1. Introduction
The primary objective of this project is to develop a robust machine learning system capable of identifying **120 distinct dog breeds** from images. We utilized the "Dog Breed Identification" dataset, a standard benchmark in computer vision.

### Key Challenges
This task presents significant challenges inherent to fine-grained image classification:
* **Inter-class Similarity:** Many distinct breeds share nearly identical visual features (e.g., distinguishing a Siberian Husky from an Alaskan Malamute).
* **Intra-class Variation:** Images of the same breed vary drastically depending on lighting, pose, age, and background.

### The Hybrid Approach
To address these complexities, we implemented a dual-pathway system:
1.  **Deep Learning:** Utilizing **NASNetLarge**, a state-of-the-art CNN pre-trained on ImageNet.
2.  **Classical Machine Learning:** Extracting 4,032-dimensional feature vectors from the CNN and feeding them into traditional classifiers (**SVM** and **Random Forest**) to benchmark performance.

---

## 2. Methodology & Data Preprocessing

### 2.1 The Dataset & Preprocessing
* **Total Classes:** 120 Dog Breeds.
* **Input Data:** RGB Color Images resized to `(331, 331)`.
* **Normalization:** Scaled pixel values to the `0-1` range for stable gradient descent.
* **Data Augmentation:** Applied horizontal flips and rotations to increase model generalization and reduce overfitting.



### 2.2 Feature Engineering (The "Layer Cut")
To satisfy the requirements for Classical ML, we performed a "layer cut" on the trained NASNetLarge architecture to transform visual data into structured numerical data.
* **The Process:** We intercepted the data at the `GlobalAveragePooling2D` layer of the base model.
* **The Output:** Every image was transformed into a flat numerical vector of **4,032 features**.
* **Persistence:** We utilized the `joblib` library to save the trained SVM and Random Forest models to Google Drive, ensuring they can be re-loaded for inference without re-running the extraction pipeline.

---

## 3. Technical Challenges & Solutions

### 3.1 Resource Optimization (OOM Errors)
We moved from loading raw NumPy arrays into RAM to a **Generator-based pipeline** using `ImageDataGenerator`. This allowed the system to stream data in small batches, preventing "Out of Memory" (OOM) crashes.

### 3.2 Session Volatility & Checkpointing
We addressed Google Colab’s ephemeral nature by mounting Google Drive and implementing a **Checkpointing Strategy**, saving the `.keras` deep learning model and `.joblib` classical models to persistent storage.

### 3.3 Transition to CPU-Only Evaluation
* **The Problem:** During final evaluation, GPU access was restricted, making full-dataset inference through NASNetLarge (88M parameters) computationally prohibitive.
* **The Solution:** We implemented a **"Representative Subset" strategy**. We extracted features from a randomized subset of the validation data, allowing for the calculation of valid performance metrics in a reasonable timeframe on a CPU.

---

## 4. Implementation Details

### 4.1 Deep Learning (CNN)
We utilized **NASNetLarge** with a Global Average Pooling layer to reduce dimensionality and a Softmax output layer for the 120-class classification.

#### Environment Setup and Data Acquisition
The project begins by connecting to Kaggle to fetch the images and labels.

**Technical Snippet:**
#### Upload kaggle.json to authenticate
if not os.path.exists('/content/kaggle.json'):
    files.upload()

#### Download and unzip the Dog Breed dataset
!kaggle competitions download -c dog-breed-identification
!unzip -q dog-breed-identification.zip -d /content/dog_data

This authenticates the session using a Kaggle API key, downloads the raw image data and labels directly into the temporary workspace

#####
Create Data Generator (Normalization & Validation Split)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

#### Training & Validation Generators (Batching)
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/content/dog_data/train',
    x_col="id",
    y_col="breed",
    subset="training",
    target_size=(331, 331),
    class_mode="categorical"
)
What this does:
- Normalization: Converts pixel values (0–255) to a 0–1 range to speed up training.
- Conveyor Belt System (Generators): Instead of loading all 10,000+ images into RAM (which would crash the system), the code loads them in small "batches" of 8 or 32 at a time.
- Resizing: Every photo is resized to 331x331 pixels to fit the NASNetLarge input requirements.


### 4.2 Classical Models (Hybrid Approach)
* **Support Vector Machine (SVM):** Utilized a **Linear Kernel**. In high-dimensional spaces (4,032 features), SVMs excel at finding the optimal hyperplane for class separation.
* **Random Forest:** An ensemble of **100 decision trees**. This served as a non-linear benchmark to evaluate if tree-based ensemble logic could outperform the linear separation of the SVM.

---

## 5. Evaluation & Results

### 5.1 Multi-Metric Performance Table
We evaluated the models using **Macro-Averaging** to ensure that performance on rare breeds was weighted equally with common breeds.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **CNN (NASNetLarge)** | ~91% | 0.92 | 0.90 | 0.91 |
| **SVM (Linear)** | ~86% | 0.87 | 0.85 | 0.86 |
| **Random Forest** | ~72% | 0.75 | 0.71 | 0.72 |

<img width="1007" height="548" alt="image" src="https://github.com/user-attachments/assets/aa314fb6-2ebe-4335-ad62-773ee9c9e879" />

<img width="741" height="149" alt="image" src="https://github.com/user-attachments/assets/3fe1c247-719d-4d71-9459-07aecd06c6a7" />


### 5.2 Comparative Analysis
Our visual analysis confirmed that the **Deep Learning CNN** maintains a lead across all four metrics. However, the **SVM** followed closely, proving that the features extracted by the NASNet "brain" are highly robust and linearly separable even without a neural classifier.

### 5.3 Error Analysis (Confusion Matrix)
The Confusion Matrix revealed that while the model is highly accurate on distinct breeds like Chihuahuas, it occasionally confuses visually similar breeds.



* **Observation:** Errors occurred primarily between breeds with similar silhouettes, such as the **Norfolk Terrier** and the **Norwich Terrier**. This suggests that fine-grained identification relies heavily on micro-features (ear shape, tail curl) captured in the high-dimensional feature vectors.

---

## 6. Conclusion
This project successfully demonstrated a comprehensive **Hybrid Machine Learning pipeline**.

1.  **Optimized Workflow:** Overcame hardware limitations (RAM/CPU) using generators and subset-based evaluation.
2.  **Feature Robustness:** Proved that features extracted via "Layer Cutting" a CNN are powerful enough to yield high accuracy in classical models.
3.  **Persistence:** Successfully created a production-ready system by saving all model types (Keras, SVM-joblib, RF-joblib) to Google Drive.

Our results confirm that while end-to-end Deep Learning is the most accurate, the **CNN + SVM hybrid** offers a compelling balance of performance and computational efficiency for fine-grained classification tasks.
