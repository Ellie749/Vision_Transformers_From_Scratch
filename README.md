# Vision Transformers from Scratch

This repository contains a from-scratch implementation of the Vision Transformer (ViT) architecture using Tensorflow. It walks through the core building blocks such as positional encoding, patch embedding, multi-head attention, and classification head.

## 📁 Project Structure
VISION_TRANSFORMERS_FROM_SCRATCH <br>
| ├── examples<br>
│ ├── src <br>
│   ├── data_pipeline <br>
│   │   └── load_dataset.py # Functions to load and preprocess datasets <br>
│   ├── model<br>
│   |    └── train.py # Functions run experiments <br>
│   ├── network<br>
│   │   ├── architecture.py # Creating and assembling ViT network architecture<br>
│   │   |   positional_encoding.py # Positional encoding for ViT <br>
│   │   └── mlp.py<br>
│   ├──  visualization<br>
│   │   └── utils.py # visualizing train metrics<br>
│   ├── weights/ # Directory to save trained model weights <br>
|   ├── experiments.ipynb # Notebook for quick experimentation<br>
|   ├── project.p # main training code<br>
| ├── application.py <br>
| ├── config.ini # Configuration file <br>
| ├── README.md # You're here!<br>
| └── requirement.txt<br>



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
```

📊 Dataset
Currently supports image classification datasets like CIFAR-100. You can modify load_dataset.py to plug in other datasets.

🧠 TODO