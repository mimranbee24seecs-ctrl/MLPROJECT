# Project Report: Dog Breed Identification Using Deep Learning and Hybrid Approaches

**By:** Filza Fatima (506785), Ammar Imran (501062)

## 1. Introduction
The primary objective of this project is to develop a robust machine learning system capable of identifying 120 distinct dog breeds from images. We utilized the "Dog Breed Identification" dataset, a standard benchmark in computer vision.

This task presents significant challenges inherent to fine-grained image classification:
1.  **Inter-class Similarity:** Many distinct breeds share nearly identical visual features (e.g., distinguishing a *Siberian Husky* from an *Alaskan Malamute*).
2.  **Intra-class Variation:** Images of the same breed vary drastically depending on lighting, pose, age (puppy vs. adult), and background clutter.

To address these complexities, we implemented a **Hybrid Approach**:
* **Deep Learning:** Utilizing **NASNetLarge**, a state-of-the-art Convolutional Neural Network (CNN) pre-trained on ImageNet, to learn hierarchical feature representations.
* **Classical Machine Learning:** Extracting high-dimensional feature vectors from the CNN and feeding them into traditional classifiers (**SVM** and **Random Forest**) to benchmark performance.

---

## 2. Methodology & Data Preprocessing

### 2.1 The Dataset
* **Total Classes:** 120 Dog Breeds.
* **Input Data:** RGB (Color) Images.
* **Preprocessing Pipeline:**
    * **Resizing:** All images were resized to `(331, 331)` pixels. This specific dimension is required by the NASNetLarge architecture to ensure spatial features are preserved correctly.
    * **Normalization:** Pixel values were scaled from `0-255` to the range `0-1`. This is critical for neural network convergence, preventing exploding gradients and ensuring stable training.
    * **Data Augmentation:** To prevent overfitting, we applied transformations (horizontal flips, rotation) to the training set, artificially expanding the dataset size.

### 2.2 Feature Engineering (The "Hybrid" Approach)
Classical algorithms (like SVMs) cannot process raw image pixels effectively; they require structured data. To satisfy the project requirement for **Classical Machine Learning**, we employed **Transfer Learning** for feature extraction:
* **The Process:** We passed every image through the frozen NASNetLarge model but intercepted the data stream at the `GlobalAveragePooling` layer, just before the final classification.
* **The Result:** We converted raw images into high-level **feature vectors** (arrays of 4,032 floating-point numbers). These numbers represent abstract concepts like "ear shape," "fur texture," or "snout length."
* These vectors served as the input `X` for our SVM and Random Forest classifiers.

---

## 3. Technical Challenges & Solutions
A significant portion of this project involved solving system-level and data-integrity issues. We faced three major hurdles:

### 3.1 Resource Optimization (The "Out of Memory" Crash)
* **The Problem:** Initially, we attempted to load the entire dataset into RAM as standard NumPy arrays (`X_train`, `y_train`). With thousands of high-resolution images ($331 \times 331 \times 3$), this required over 16GB of RAM, causing immediate environment crashes (OOM Errors).
* **The Solution:** We refactored the pipeline to use TensorFlow’s `ImageDataGenerator`. Instead of loading all data at once, this approach uses **dynamic loading**, fetching and processing images in small batches (e.g., 16 images at a time) only when the GPU requests them. This kept RAM usage low and stable.

### 3.2 Data Mismatch & Integrity
* **The Problem:** During the initial data loading phase, the generator reported "0 images found," despite the files being present. Upon investigation, we discovered a discrepancy between the CSV labels and the actual filenames. The CSV listed IDs as `000bec18...`, while the files on disk were `000bec18....jpg`.
* **The Solution:** We implemented a Pandas preprocessing script to append the `.jpg` extension to every ID in the dataframe dynamically (`df["id"].apply(...)`), ensuring a 100% match between labels and files.

### 3.3 Session Volatility & Persistence
* **The Problem:** Google Colab sessions are ephemeral. We experienced session disconnects which wiped out temporary variables (such as the training history and the `kaggle.json` credentials), making it impossible to plot training curves post-crash.
* **The Solution:** We implemented a rigorous **Checkpointing Strategy**.
    1.  We mounted **Google Drive** (`drive.mount`) to create persistent storage.
    2.  We configured the model to save `.h5` files directly to Drive after training.
    3.  We developed a "Recovery Block" of code that could reload the saved model and re-initialize the data generators without needing to re-train from scratch, allowing us to generate evaluation metrics even after a browser crash.

---

## 4. Implementation Details

### 4.1 Deep Learning Architecture
We utilized **NASNetLarge**, an architecture discovered by Google's Neural Architecture Search (NAS).
* **Feature Extractor:** The base model layers were **frozen**, ensuring we retained the generic visual knowledge learned from ImageNet.
* **Pooling Strategy:** We used **Global Average Pooling** rather than Flattening. This reduces the tensor dimensions by averaging feature maps, which drastically reduces the number of parameters and minimizes the risk of overfitting.
* **Classifier:** A final Dense layer with 120 neurons and `softmax` activation provided the probability distribution across the breeds.

### 4.2 Classical Models
1.  **Support Vector Machine (SVM):** Configured with a `linear` kernel. SVMs proved effective in high-dimensional spaces (4,032 features), finding a hyperplane that separates breeds with a clear margin.
2.  **Random Forest:** An ensemble of 100 decision trees. This model struggled slightly compared to SVM, likely due to the "Curse of Dimensionality," where decision trees find it hard to split effectively on 4,000+ sparse features.

---

## 5. Evaluation & Results

### 5.1 Performance Comparison

| Model | Accuracy | Analysis |
| :--- | :--- | :--- |
| **Deep Learning (NASNetLarge)** | **~90%+** | **Superior Performance.** The end-to-end nature of the CNN allows it to capture subtle, non-linear dependencies between pixels that classical models miss. |
| **SVM (Linear Kernel)** | **~85%** | **Strong Contender.** The high accuracy proves that the features extracted by NASNet are robust and linearly separable. |
| **Random Forest** | **~75%** | **Weakest.** Random Forests generally perform better on tabular data than on high-dimensional dense vectors derived from images. |

### 5.2 Error Analysis
Using the **Confusion Matrix**, we identified that the model struggles most with breeds that are virtually identical, such as the *Norfolk Terrier* vs. *Norwich Terrier*. However, it achieved near-perfect precision on distinct breeds like *Pugs* and *Samoyeds*.

---

## 6. Conclusion
This project successfully demonstrated the power of **Transfer Learning** in solving complex computer vision tasks with limited data.
1.  We successfully migrated from a memory-heavy workflow to an optimized **Generator-based pipeline**.
2.  We overcame significant data formatting and session persistence challenges.
3.  Our evaluation confirms that while Deep Learning yields the best results, a Hybrid approach (CNN + SVM) is a viable and computationally efficient alternative for deployment.
