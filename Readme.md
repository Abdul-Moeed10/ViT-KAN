# ViT-KAN: Vision Transformer with Kolmogorov–Arnold Networks for Human Action Recognition

**ViT-KAN** is a hybrid deep learning architecture designed to improve the performance and efficiency of human action recognition tasks. This model integrates a Vision Transformer (ViT-B/16) backbone—pre-trained with SWAG (Stochastic Weight Averaging Gaussian) weights on ImageNet-21k—for extracting robust spatial features from video frames or images. These features are passed through a classification head constructed using Kolmogorov–Arnold Networks (KANs), which model nonlinear relationships through learnable spline-based functions. This combination allows for accurate classification while maintaining flexibility and interpretability.

The model is evaluated on two benchmark datasets—Penn Action and Human Action Recognition (HAR)—and demonstrates competitive accuracy while being computationally efficient.

---

## 📚 Datasets

### 🏸 Penn Action Dataset
The **Penn Action dataset** contains 2326 annotated video sequences spanning 15 different human actions, such as tennis serve, squat, pull-up, and strum guitar. Each sequence is accompanied by joint annotations. Video frames are extracted and resized to 384×384 resolution, normalized using ImageNet statistics, and processed into temporal stacks.

- [Dataset Link](https://dreamdragon.github.io/PennAction/)
- Preprocessing includes:
  - Resizing frames to 384x384
  - Frame sampling to 32 per video
  - Normalization using ViT-compatible ImageNet statistics

### 🧍‍♂️ Human Action Recognition (HAR) Dataset
The **HAR dataset**, sourced from Kaggle, consists of labeled images of diverse human activities. Each image corresponds to a class label identifying the performed action. This dataset is used to test single-frame recognition performance.

- [Kaggle Link](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)
- Images are resized to 384×384 and normalized before training.

---

## 🚀 How to Use the Code

### 🛠 Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- pykan
- tqdm
- scikit-learn
- matplotlib

### 📥 Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/vit-kan.git
    cd vit-kan
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install KAN library**
    ```bash
    git clone https://github.com/KindXiaoming/pykan.git
    pip install ./pykan
    ```

---

### 🧪 Running on Penn Action Dataset

1. Download the dataset from [Penn Action website](https://dreamdragon.github.io/PennAction/).
2. Upload `data.zip` to Google Drive and mount it in Google Colab.
3. Extract and preprocess the frames.
4. Run the training script in Colab to begin model training.

---

### 🧪 Running on HAR Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) and unzip.
2. Ensure the dataset folder has the proper CSV file and training images.
3. Run the training script with the image dataset path specified.

---

## 📊 Results & Performance

- Achieved **98.9% accuracy** on Penn Action dataset.
- Achieved **96.5% accuracy** on HAR dataset.
- Demonstrated strong generalization, fast convergence, and smooth loss minimization across both datasets.

