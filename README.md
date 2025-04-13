# 🚨 Toxic Feedback Classification

A robust multi-label classification system built to detect toxic behaviors in user feedback across six categories: **toxic**, **abusive**, **vulgar**, **menace**, **offense**, and **bigotry**. The final prediction focuses on identifying whether a piece of content is offensive.

<p align="center">
  <img src="assets/cover.png" width="600" alt="Toxic Classification Banner">
</p>

---

## 📁 Project Structure

- `model1_implementation.ipynb`: Baseline and classical ML models using TF-IDF and Logistic Regression.
- `model2_implementation.ipynb`: Advanced models using LSTM, GRU, and Transformer-based architectures (BERT/XLM).
- `train.csv`, `test.csv`: Cleaned and preprocessed datasets.
- `assets/`: Visuals and plots used in this README.

---

## 📊 Dataset Overview

| Feature         | Description                          |
|-----------------|--------------------------------------|
| feedback_text   | User-generated content               |
| toxic           | Binary label for toxicity            |
| abusive         | Binary label for abusive content     |
| vulgar          | Binary label for vulgar language     |
| menace          | Binary label for threatening content |
| offense         | **Target** binary label              |
| bigotry         | Binary label for bigoted remarks     |

All texts have been normalized, lemmatized, and translated into English for consistency.

---

## 🧠 Models Overview

### ✅ Model 1: Baseline Classifier
- **Vectorizer**: TF-IDF (unigram & bigram)
- **Classifier**: Logistic Regression
- **Evaluation**: F1 Score on 'offense' label

<p align="center">
  <img src="assets/model1_tfidf_lr.png" width="450" alt="TF-IDF + Logistic Regression">
</p>

---

### 🤖 Model 2: Deep Learning & Transformers
- **RNNs**: LSTM & GRU
- **Transformer**: BERT / XLM-RoBERTa (fine-tuned)
- **Embeddings**: Pre-trained word embeddings (GloVe / FastText)

<p align="center">
  <img src="assets/model2_architecture.png" width="600" alt="Model 2 Architecture Diagram">
</p>

---

## 📈 Results

| Model                | F1 Score (Offense) |
|---------------------|-------------------|
| Logistic Regression | 0.84              |
| LSTM                | 0.87              |
| GRU                 | 0.86              |
| BERT                | 0.91              |
| XLM-RoBERTa         | **0.92**          |

<p align="center">
  <img src="assets/f1_comparison_chart.png" width="500" alt="F1 Score Comparison">
</p>

---

## 🛠️ Usage

### ⚙️ Setup Environment
```bash
pip install -r requirements.txt
