# Arabic Sentiment Analysis (ENCS3340 Machine Learning Project) 💬🇵🇸

Birzeit University — Department of Electrical & Computer Engineering 🎓  
**ENCS3340 Artificial Intelligence** | First Semester **2025–2026** 🧠  
**Machine Learning Project** 
Topic: **Arabic Sentiment Analysis** ✨

---

## Overview 🔎
This project develops and evaluates multiple machine learning models for **Arabic sentiment classification** on social media text. The dataset is labeled into three classes:

- **POS** (Positive) 😊  
- **NEG** (Negative) 😡  
- **OBJ** (Objective/Neutral) 😐  

Arabic sentiment analysis is challenging due to **dialectal variation**, **spelling noise**, **elongation**, and possible **class imbalance** ⚠️.  
This repository implements the full ML pipeline: EDA → preprocessing → feature engineering → training → tuning → evaluation ✅

---

## Objectives 🎯
- Explore the dataset structure and label distribution 📊
- Clean and normalize Arabic text 🧹
- Build multiple feature representations 🧩:
  - TF-IDF (word n-grams) 📝
  - TF-IDF (character n-grams) 🔤
  - Word embeddings (Word2Vec & FastText) 🧬
  - Handcrafted Arabic-specific features 🛠️
- Train and compare classifiers 🤖:
  - Decision Tree 🌳
  - Random Forest 🌲🌲
  - Naïve Bayes 🧮
  - Artificial Neural Network (MLP) 🧠
- Use **60% / 20% / 20%** train/val/test split ✂️
- Evaluate using: **Accuracy, Precision, Recall, F1-score** + **Confusion Matrix** 📈

---

## Dataset 📦
The dataset is provided as:

- `dataset.txt`  
Each row format:  
`<tweet_text>\t<label>`

Labels are normalized to handle variants:
- `NEUTRAL`, `Neutral`, `Natural` → `OBJ` ✅

> ⚠️ The dataset may include URLs, emojis, informal expressions, and dialect vocabulary.

---

## Environment & Requirements 🧰

### Libraries Used 📚
- Data: `pandas`, `numpy`
- NLP/Arabic: `nltk`, `pyarabic`, `emoji`, `re`
- ML & Features: `scikit-learn`, `gensim`, `scipy`
- Visualization: `matplotlib`, `seaborn`
- Utilities: `pickle`

### Install ⬇️
```bash
pip install pandas numpy nltk pyarabic emoji gensim scipy scikit-learn matplotlib seaborn

---
