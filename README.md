# Text Emotion Recognition

A comprehensive comparative analysis of various deep learning approaches for text emotion recognition.

## Overview

This project evaluates and compares different deep learning architectures for text emotion recognition:

* **Attention-based Bidirectional Long Short-Term Memory (BiLSTM)**
* **Convolutional Neural Network with Bidirectional Gated Recurrent Unit (CNN-BiGRU)**
* **Transformer-based models** including:
  * Multi-Head Attention (MHA)
  * Multi-Query Attention (MQA)
  * Grouped-Query Attention (GQA)
  * Multi-Head Latent Attention (MLA)

## Project Structure

```
📦 
├── dataset/
│   ├── models/
│   ├── result/
│   ├── wassa_data/
│   ├── combined_emotion_kaggle_kush...
│   ├── text_emotion.csv
│   └── wassa_combined_data.csv
├── .gitignore
├── common_utils.py
├── part_1_data_preprocessing.ipynb
├── part_2_bigru.ipynb
├── part_2_BILSTM.ipynb
├── part_2_cnn_bigru.ipynb
├── part_2_transformer.ipynb
├── README.md
├── SC4001_Report.pdf
├── transformer.py
└── wassa_combined_data.csv
```

## Dataset

The project uses the WASSA dataset, which contains Twitter data labeled with emotions (anger, fear, joy, and sadness) and intensity scores. The dataset is split into training, validation, and testing sets with a total of 7,102 samples and a vocabulary size of 19,117 words.

## Key Components

### Data Preprocessing (`part_1_data_preprocessing.ipynb`)
- Loads and processes the text data
- Implements GloVe embeddings (6 billion tokens, 100-dimensional vectors)
- Creates embedding matrices for the vocabulary

### Model Implementations
1. **BiLSTM Implementation** (`part_2_BILSTM.ipynb`)
   - Baseline BiLSTM model
   - Attention-enhanced BiLSTM model

2. **BiGRU Implementation** (`part_2_bigru.ipynb`)
   - Baseline BiGRU model

3. **CNN-BiGRU Implementation** (`part_2_cnn_bigru.ipynb`)
   - Hybrid architecture combining CNN and BiGRU layers

4. **Transformer Models** (`part_2_transformer.ipynb` and `transformer.py`)
   - Implements various attention mechanisms:
     - Multi-Head Attention (MHA)
     - Multi-Query Attention (MQA)
     - Grouped-Query Attention (GQA)
     - Multi-Head Latent Attention (MLA)

### Utilities (`common_utils.py`)
Contains helper functions used across the project.

## Results

Our experiments demonstrate that the CNN-BiGRU architecture achieves superior performance with an F1 score of 0.8855, outperforming both the Attention-based BiLSTM (0.8657) and Transformer-based models, with the best Transformer variant (GQA) reaching an F1 score of 0.8649.

| Model | Test F1 Score | 
|-------|---------------|
| CNN-BiGRU | 0.8855 |
| Attention-BiLSTM | 0.8657 | 
| Transformer (GQA) | 0.8649 | 
| Transformer (MHA) | 0.8625 | 
| Transformer (MQA) | 0.8599 | 
| Transformer (MLA) | 0.8377 | 

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/woonyee28/DLNN_NLP.git
cd DLNN_NLP
```

### Running the Models

1. First, run the data preprocessing notebook:
```bash
jupyter notebook part_1_data_preprocessing.ipynb
```

2. Then, run the model notebooks:
```bash
jupyter notebook part_2_BILSTM.ipynb 
jupyter notebook part_2_cnn_bigru.ipynb 
jupyter notebook part_2_transformer.ipynb  
```

## Detailed Report

For a comprehensive analysis of the models and findings, please refer to the [full report](SC4001_Report.pdf).

## Contributors

- Ng Woon Yee 
- Lye Jin Kai 
- Won Tian Cong, Adriel
