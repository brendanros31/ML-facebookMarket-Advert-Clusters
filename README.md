# Facebook Marketplace Listing Analysis – Thailand 🇹🇭

This project analyzes listing data from **Facebook Marketplace in Thailand** to uncover how users engage with posts. 

The goal is to identify the most **impactful listing strategies** by understanding- what makes listings more effective?.

We use **reactions, comments, shares**, and **engagement patterns** from the Listings, and apply unsupervised and supervised learning techniques to achive the desired outcome.

---

## Project Overview

We explore a dataset of product listings and their engagement metrics to answer:

- What kinds of posts get more reactions, shares, and comments?
- Are there identifiable **clusters** of listing types based on engagement?
- Can we predict listing effectiveness using classification models?
- What features influence listing performance the most?

---

## Methods & Techniques Used

### 🧼 Data Preprocessing
- Handled outliers in the data for better model performance
- Feature engineering (e.g. total positive/negative engagement score)
- Standardization of numeric features

### 📊 Exploratory Data Analysis
- Visualized distribution of shares, reactions, comments
- Category-wise engagement insights

### 📈 Clustering with K-Means
- Grouped listings based on engagement features
- Used **Elbow Method** and **Silhouette Score** to determine optimal clusters
- Visualized clusters using **PCA** (Principal Component Analysis)

### 🤖 Classification Models
- Trained models:
  - **Logistic Regression**
  - **Random Forest Classifier**
- Evaluated using **accuracy**, **precision**, **recall**, and **F1 score**

### 📊 Dimensionality Reduction
- Used **PCA** for visualization and performance tuning
- Explored feature importance

---

## How to Use
```bash
git clone https://github.com/brendanros31/ML-facebookMarket-Listing-Clusters.git

cd ML-facebookMarket-Listing-Clusters

pip install -r requirements.txt
jupyter notebook main.ipynb
```

---

## Project Structure
```
data/
  raw/
    FBMarketplace_Thai.csv

src/
  data_loader.py
  evaluate.py
  features.py
  model.py
  utils.py

EDA.ipynb
main.ipynb
config/config.yaml
```