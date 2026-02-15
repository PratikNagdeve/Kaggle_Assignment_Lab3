# Kaggle_Assignment_Lab3 

# ⚡ ML Lab – Performance Comparison of CPU, GPU and TPU using TensorFlow

## 📌 Aim

To implement a deep learning model and compare the training time on CPU, GPU, and TPU using TensorFlow in a cloud environment.

---

## 🎯 Objectives

* Detect and initialize available hardware accelerator (CPU/GPU/TPU)
* Train the same model on different hardware
* Measure and compare execution time
* Understand parallel processing capabilities of GPU and TPU

---

## 🛠️ Tools & Technologies Used

* Python
* TensorFlow / Keras
* Google Colab (TPU support) / Kaggle
* NumPy
* Time module (for performance measurement)

---

## 💾 Dataset

**MNIST Handwritten Digit Dataset**

* 60,000 training images
* 10,000 testing images
* 10 classes (0–9)

Loaded using:

```python
tf.keras.datasets.mnist
```

---

## 🧠 Methodology

### 1️⃣ Hardware Detection

The system automatically detects available hardware:

* TPU using `TPUClusterResolver`
* GPU using `tf.config.list_physical_devices('GPU')`
* Falls back to CPU if none available

---

### 2️⃣ Distribution Strategy

* `TPUStrategy` → for TPU
* `MirroredStrategy` → for GPU
* Default strategy → for CPU

---

### 3️⃣ Model Architecture

A simple neural network:

* Flatten layer
* Dense (ReLU)
* Output layer (Softmax)

---

### 4️⃣ Training

* Same model
* Same dataset
* Same number of epochs

Training time measured using:

```python
start = time.time()
end = time.time()
```

---

## 📊 Observations

| Hardware | Replicas        | Training Time                |
| -------- | --------------- | ---------------------------- |
| CPU      | 1               | Higher                       |
| GPU      | 1               | Faster than CPU              |
| TPU      | 8 (or VM based) | Fastest for large batch size |

---

## ✅ Result

* Successfully detected and initialized hardware accelerator
* Model trained on available device
* Execution time measured and compared
* Demonstrated that parallel processors significantly reduce training time for large workloads

---

## 📈 Conclusion

* GPU provides faster computation than CPU due to parallel processing.
* TPU gives maximum performance when:

  * Batch size is large
  * Model is trained using `TPUStrategy`
* Hardware acceleration is essential for large-scale deep learning.

---

## 🌍 Real-World Applications

This workflow is used in:

* Large-scale deep learning training
* Federated learning systems
* MLOps pipelines
* Research environments

---

## 📷 Sample Output

```
Device: TPU
Replicas: 8
Training Time: XX seconds
```

---

## 👨‍💻 Author

**Pratik Nagdeve**
M.Tech Artificial Intelligence
MANIT Bhopal

---

## 🔗 Open in Google Colab

*(Add your Colab notebook link here)*

```md
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PASTE_YOUR_COLAB_LINK)
```

---
