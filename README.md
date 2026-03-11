# Naive Bayes Text Classification

SMS Spam Detection and BBC News Classification

## Overview

This project implements a **Naive Bayes Classifier from scratch** to perform text classification on two datasets:

1. **SMS Spam Collection Dataset** – classify messages as **spam** or **ham**.
2. **BBC News Dataset** – classify news articles into categories such as **business, politics, sport, tech, and entertainment**.

Two Naive Bayes variants are implemented:

* **Binary (Bernoulli) Naive Bayes** – used for SMS spam detection
* **Multinomial Naive Bayes** – used for BBC news classification

The implementation includes:

* Text preprocessing
* Vocabulary creation
* Training Naive Bayes model
* Prediction and evaluation
* Confusion matrix visualization
* Analysis of top indicative words

---

# Project Structure

```
├── naive_bayes_classifier.ipynb
├── README.md
├── SMSSpamCollection
├── BBC News Summary/
│   └── News Articles/
│       ├── business/
│       ├── entertainment/
│       ├── politics/
│       ├── sport/
│       └── tech/
```

---

# Datasets

## 1. SMS Spam Collection (UCI)

A dataset containing SMS messages labeled as:

* **ham** – legitimate messages
* **spam** – unwanted promotional messages

Download Source:
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Example:

| Label | Message                                           |
| ----- | ------------------------------------------------- |
| ham   | Ok lar... Joking wif u oni...                     |
| spam  | Free entry in a weekly competition to win tickets |

---

## 2. BBC News Dataset

A dataset of news articles categorized into:

* business
* entertainment
* politics
* sport
* tech

Each article is stored as a text file within its category folder.

---

# Libraries Used

```
pandas
numpy
re
collections
sklearn
matplotlib
seaborn
requests
zipfile
os
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

# Text Preprocessing

The following preprocessing steps are applied:

1. Convert text to lowercase
2. Remove punctuation and numbers
3. Remove extra spaces
4. Tokenize text into words

Example:

Input:

```
"Win FREE tickets now!!!"
```

Output tokens:

```
["win", "free", "tickets", "now"]
```

---

# Naive Bayes Implementation

The classifier computes:

### Prior Probability

[
P(C) = \frac{\text{documents in class C}}{\text{total documents}}
]

### Likelihood

**Multinomial Model**

[
P(w|C) = \frac{count(w,C)+\alpha}{total\ words\ in\ C + \alpha|V|}
]

**Binary Model**

[
P(w|C) = \frac{docs\ containing\ w + \alpha}{docs\ in\ C + 2\alpha}
]

### Posterior Score

[
score(C) = log P(C) + \sum log P(w|C)
]

The class with the **highest score** is selected.

---

# Training and Evaluation

## SMS Spam Classification

Model: **Binary Naive Bayes**

Steps:

1. Split dataset (80% training, 20% testing)
2. Train classifier
3. Evaluate accuracy
4. Test with sample messages

Example predictions:

```
"Win free tickets now" -> spam
"Are you coming to the meeting?" -> ham
```

---

## BBC News Classification

Model: **Multinomial Naive Bayes**

Steps:

1. Stratified train-test split
2. Train classifier
3. Predict categories
4. Generate confusion matrix

Confusion matrix visualization is plotted using **Seaborn heatmap**.

---

# Indicative Words Analysis

The project also identifies **Top 10 words most indicative of each class** using a likelihood ratio:

[
score = log(P(w|C)) - log(P(w|NotC))
]

Example:

### SMS Spam

```
free, win, prize, claim, urgent
```

### BBC News (Sport)

```
team, match, season, league
```

---

# How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/naive-bayes-text-classification.git
```

Navigate to project:

```bash
cd naive-bayes-text-classification
```

Run the notebook:

```bash
jupyter notebook naive_bayes_classifier.ipynb
```

or open it in **VS Code / Google Colab**.

---

# Results

| Dataset  | Model                   | Accuracy      |
| -------- | ----------------------- | ------------- |
| SMS Spam | Binary Naive Bayes      | High accuracy |
| BBC News | Multinomial Naive Bayes | High accuracy |

Performance may vary slightly depending on preprocessing and dataset split.

---

# Visualizations

* Confusion Matrix for BBC News
* Top Indicative Words per Class

---

# Key Concepts Demonstrated

* Natural Language Processing (NLP)
* Text preprocessing
* Naive Bayes classification
* Laplace smoothing
* Multinomial vs Binary Naive Bayes
* Model evaluation

---

# Future Improvements

* Add stopword removal
* Add TF-IDF features
* Compare with Scikit-learn Naive Bayes
* Add Precision, Recall, F1-score
* Deploy as a web app

---

# Author

Sanjana

Machine Learning / NLP Project
