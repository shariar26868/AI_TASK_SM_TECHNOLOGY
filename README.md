# 🚨 Toxic Feedback Classification

A robust multi-label classification system built to detect toxic behaviors in user feedback across six categories: **toxic**, **abusive**, **vulgar**, **menace**, **offense**, and **bigotry**. The final prediction focuses on identifying whether a piece of content is offensive.

<p align="center">
  <img src="https://github.com/user-attachments/assets/27ed2ccb-c48e-4369-91e8-69464aa7223d" width="600" alt="download">
</p>


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
| toxic           | **Target** Binary label for toxicity |
| abusive         | Binary label for abusive content     |
| vulgar          | Binary label for vulgar language     |
| menace          | Binary label for threatening content |
| offense         | binary label              |
| bigotry         | Binary label for bigoted remarks     |

All texts have been normalized, lemmatized, and translated into English for consistency.

---

## 🧠 Models Overview

### ✅ Model 1: Baseline Classifier
- **Vectorizer**: TF-IDF (unigram & bigram)
- **Classifier**: Logistic Regression
- **Evaluation**: F1 Score on 'toxic' label

<p align="center">
  <img src="https://github.com/user-attachments/assets/f927c3ab-43ea-44b4-a8b1-81336daa93e0" alt="download" width="600">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6cc8c429-a555-4981-b666-9e659f20a0ff" alt="download" width="600">
</p>


### 🤖 Model 2: Deep Learning & Transformers
- **RNNs**: LSTM & GRU
- **Transformer**: BERT  (fine-tuned)

<p align="center">
  <img src="https://github.com/user-attachments/assets/cce1a8b4-6c10-4140-ab5f-6f300f82f8f5" alt="image" width="600">
</p>


## 📈 Results

| Model                | F1 Score (Offense) |
|---------------------|-------------------|
| Logistic Regression | 0.84              |
| LSTM                | 0.87              |
| GRU                 | 0.86              |
| BERT                | 0.91              |


## ✅ Conclusion

This project demonstrates a scalable and effective approach to detecting offensive and toxic content in user feedback. By combining classical machine learning methods with advanced deep learning and Transformer-based models, we achieved high accuracy in offense classification.

Key takeaways:

- Cleaned and normalized multilingual data improves model generalization.
- TF-IDF with Logistic Regression offers a strong baseline.
- Deep learning models (LSTM/GRU) enhance context awareness.
- Transformer models like BERT outperform traditional methods, with XLM achieving the best F1 score.

This pipeline can serve as a foundation for real-time content moderation systems across platforms, ensuring safer and more inclusive digital spaces.




