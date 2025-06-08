# 🧬 Structural Protein Sequences Classification

This project explores the application of Natural Language Processing (NLP) techniques to classify structural protein sequences. By treating protein sequences like natural language, we leverage deep learning models specifically BiLSTM to identify and classify structural types based on sequence patterns.

## 📌 Project Overview

Proteins are composed of sequences of amino acids, and their structure plays a critical role in their function. This project applies NLP techniques, such as tokenization and sequence modeling, to classify protein sequences into their structural categories.

## 🧠 Model Architecture

-   **Input**: Raw protein sequences
-   **Text Preprocessing**: Tokenization & Embedding
-   **Model**: Bidirectional LSTM (BiLSTM)
-   **Output**: Multiclass classification of protein structure types

## 📊 Evaluation Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 42.55% |
| Precision | 41.19% |
| Recall    | 42.55% |
| F1-Score  | 39.75% |

> 🔑 Note: The model is still under development and may benefit from further hyperparameter tuning, more data, and advanced embeddings (e.g., ESM, ProtBert).

## 🧪 Dataset

-   **Source**: _https://www.kaggle.com/datasets/shahir/protein-data-set_
-   **Classes**: _32_
-   **Size**: _46000_

## 🛠️ Tech Stack

-   Python
-   NumPy
-   Pandas
-   Scikit-learn
-   Keras
-   TensorFlow

## 🚀 How to Run (using Anaconda)

1. **Clone the repository**

```bash
git clone https://github.com/naman22a/Structural-Protein-Sequences-using-NLP
cd Structural-Protein-Sequences-using-NLP
```

2. **Create and activate a conda environment**

```bash
conda create -n protein-nlp python=3.10 -y
conda activate protein-nlp
```

3. **Install dependencies**

```bash
conda env create -f environment.yml
```

4. **Run the training script**

```bash
jupyter notebook
```

## 🧭 Future Work

-   Improve classification performance using pretrained protein models (e.g., ESM, ProtT5)
-   Experiment with attention-based models (e.g., Transformer, BERT)
-   Perform interpretability analysis on model predictions

## 📫 Stay in touch

-   Author - [Naman Arora](https://namanarora.vercel.app)
-   Twitter - [@naman_22a](https://twitter.com/naman_22a)

## 🗒️ License

Structural Protein Sequences Classification is [GPL V3](./LICENSE)
