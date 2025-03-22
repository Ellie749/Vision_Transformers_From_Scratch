# Vision Transformers from Scratch

This repository contains a from-scratch implementation of the Vision Transformer (ViT) architecture using Tensorflow. It walks through the core building blocks such as positional encoding, patch embedding, multi-head attention, and classification head.

## 📁 Project Structure
VISION_TRANSFORMERS_FROM_SCRATCH/ │ ├── src/ │ ├── data_pipeline/ │ │ └── load_dataset.py # Functions to load and preprocess datasets │ ├── model/ │ │ └── network/ │ │ └── positional_encoding.py # Positional encoding for ViT │ └── visualization/ # (Optional) Visual tools or helpers │ ├── weights/ # Directory to save trained model weights ├── examples/ # Example training/inference scripts ├── application.py # Main entry point ├── project.py # Project-specific logic (e.g., training loop) ├── experiments.ipynb # Notebook for quick experimentation ├── config.ini # Configuration file ├── README.md # You're here!

