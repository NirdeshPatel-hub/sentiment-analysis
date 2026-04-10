# 💬 Sentiment Analysis — Product Review NLP Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-88.4%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A production-ready NLP sentiment analysis engine that classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** using a TF-IDF + Naïve Bayes pipeline — achieving **88.4% accuracy** on 50,000+ reviews with a custom 1,200-term domain lexicon.

---

## 🚀 Live Demo

Run locally with one command:
```bash
streamlit run app.py
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **88.4%** |
| Precision (Macro) | **87.1%** |
| Recall (Macro) | **86.8%** |
| F1-Score (Macro) | **86.9%** |
| Reviews Classified | **50,000+** |

---

## 🏗️ Project Structure

```
sentiment-analysis/
├── app.py               # Streamlit web app
├── train.py             # Model training pipeline
├── model.py             # ML model & preprocessing
├── utils.py             # Helper functions & custom lexicon
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ How It Works

```
Raw Review Text
      │
      ▼
Text Preprocessing (lowercase, remove HTML, punctuation, stopwords)
      │
      ▼
Custom Domain Lexicon Augmentation (1,200 terms)
      │
      ▼
TF-IDF Vectorisation (max_features=50,000, ngram_range=(1,2))
      │
      ▼
Multinomial Naïve Bayes Classifier
      │
      ▼
Sentiment Label: Positive / Neutral / Negative
```

---

## 🛠️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/NirdeshPatel-hub/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python train.py

# 5. Launch the web app
streamlit run app.py
```

---

## 📦 Key Technologies

- **Python 3.9+** — Core language
- **scikit-learn** — TF-IDF + Naïve Bayes pipeline
- **NLTK** — Tokenization, stopword removal, stemming
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Interactive web dashboard
- **Matplotlib / Seaborn** — Visualisation

---

## 🎯 Key Features

- ✅ Real-time sentiment prediction via Streamlit UI
- ✅ Batch CSV upload for bulk classification
- ✅ Custom 1,200-term domain lexicon for improved accuracy
- ✅ Confusion matrix + classification report visualisation
- ✅ Misclassification analysis to identify edge cases
- ✅ 15% improvement over baseline TF-IDF model

---

## 👨‍💻 Author

**Nirdesh Patel** — Data Science & ML | DRDO Intern  
📧 nirdeshpatel280@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/nirdeshpatel) | [GitHub](https://github.com/NirdeshPatel-hub)
