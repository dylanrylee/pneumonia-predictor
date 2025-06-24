# Pneumonia Xray Detector

A convolutional neural network (CNN)–based classifier that analyzes chest X-ray images to detect pneumonia, with built-in Grad-CAM explainability and a Streamlit web interface for interactive testing.

---

## 🚀 Features

- **CNN Architecture** with Batch Normalization and Dropout for robust pneumonia classification  
- **Data Preprocessing Pipeline**:  
  - Resize to 224×224 grayscale  
  - Intensity normalization & inversion (white-on-black)  
- **Model Training**  
  - Early stopping and learning-rate scheduling  
  - Achieves ~94% test accuracy  
- **Explainability** via Grad-CAM heatmaps highlighting activation regions  
- **Streamlit Web App** for uploading X-rays, viewing predictions & confidence scores, and overlaying Grad-CAM visualizations  

---

## Dataset
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
