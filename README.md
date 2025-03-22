# Vision Transformers from Scratch

This repository contains a from-scratch implementation of the Vision Transformer (ViT) architecture using Tensorflow. It walks through the core building blocks such as positional encoding, patch embedding, multi-head attention, and classification head.

## 📁 Project Structure
VISION_TRANSFORMERS_FROM_SCRATCH
| ├── examples
│ ├── src 
│   ├── data_pipeline 
│   │   └── load_dataset.py # Functions to load and preprocess datasets 
│   ├── model
│   |    └── train.py # Functions run experiments 
│   ├── network
│   │   ├── architecture.py # Creating and assembling ViT network architecture
│   │   |   positional_encoding.py # Positional encoding for ViT 
│   │   └── mlp.py
│   ├──  visualization
│   │   └── utils.py # visualizing train metrics
│   ├── weights/ # Directory to save trained model weights 
|   ├── experiments.ipynb # Notebook for quick experimentation
|   ├── project.p # main training code
| ├── application.py 
| ├── config.ini # Configuration file 
| ├── README.md # You're here!
| └── requirement.txt



## 🚀 Features

- 🔢 Patch embedding and positional encoding
- 🧠 Transformer blocks (multi-head attention, MLP)
- 📦 Data pipeline for loading and preprocessing image datasets
- 🧪 Jupyter Notebook for testing components interactively
- 🏷️ Modular codebase for easy extension

## 🛠️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt


📊 Dataset
Currently supports image classification datasets like CIFAR-10. You can modify load_dataset.py to plug in other datasets.

🧠 TODO