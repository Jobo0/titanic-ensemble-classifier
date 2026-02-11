# Titanic Survival Prediction: Ensemble Learning Pipeline

An end-to-end Machine Learning pipeline that predicts passenger survival on the Titanic with **79.4% accuracy** (Top 5% kaggle participants leaderboard). 

This project moves beyond standard tutorials by implementing a **Voting Ensemble** (XGBoost, Random Forest, Logistic Regression) and engineering custom features to capture passenger group dynamics.

---

## Key Results

The ensemble model significantly outperformed individual baselines by correcting for the high variance of tree-based models.

| Model Architecture | CV Accuracy | Kaggle Score (Public LB) |
| :--- | :--- | :--- |
| Random Forest | 85.4% | - |
| XGBoost | 86.7% | 0.779 |
| Logistic Regression | 86.7% | - |
| **Voting Ensemble (Final)** | **86.5%** | **0.794 (Rank ~600)** |

## The Approach

### 1. Advanced Feature Engineering
* **Family Survival Groups:** Instead of treating passengers as isolated data points, I grouped them by Surname/Ticket. If a family member survived, the probability of others surviving increases drastically. This single feature boosted model confidence by ~3%.
* **Custom Binning:** Implemented "Human-Logic" binning for Age (Child vs. Adult) and Family Size to capture non-linear survival rates. Binning is not applicable to decision tree models, but greatly improved the Logistic Regression model's performance. 

### 2. Model Architecture
I used a **Voting Classifier** to combine three distinct "expert" models:
* **XGBoost:** Captures complex, non-linear interactions.
* **Random Forest:** Provides stability and handles outliers well.
* **Logistic Regression:** Acts as a linear baseline to prevent overfitting on noise (regularized with L2/Ridge).

---

## Installation

This project uses **Conda** for dependency management to ensure reproducibility.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/jobo0/titanic-ensemble-classifier.git](https://github.com/jobo0/titanic-ensemble-classifier.git)
   cd titanic-ensemble-classifier
   ```

2. **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
    
3. **Activate the environment:**
    ```bash
    conda activate titanic-env
    ```

## Useage

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Open the notebooks:**

- `notebooks/01_Exploratory_Analysis.ipynb`: Detailed EDA, feature correlation heatmaps, and initial experiments.
- `notebooks/02_Final_Pipeline.ipynb`: The clean, production-ready pipeline that generates the final submission.

## Project Structure

```plaintext
titanic-ensemble/
│
├── data/                  # Raw and processed CSV files
├── notebooks/             # Jupyter notebooks for analysis and modeling
│   ├── 01_Exploratory_Analysis.ipynb
│   └── 02_Final_Pipeline.ipynb
│
├── src/                   # Source code for modular scripts
│   ├── preprocessing.py   # Custom transformers & binning logic
│   └── ensemble.py        # Voting classifier setup
│
├── models/                # Serialized models (.pkl)
├── environment.yml        # Conda environment definition
└── README.md              # Project documentation
```
