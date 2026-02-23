# Titanic Survival Prediction: Ensemble Learning Pipeline

An end-to-end Machine Learning pipeline that predicts passenger survival on the Titanic with **79.4% accuracy** (Top 5% kaggle participants leaderboard). 

This project moves beyond standard tutorials by implementing a **Voting Ensemble** (XGBoost, Random Forest, Logistic Regression) and engineering custom features to capture passenger group dynamics.

---

## Key Results

The ensemble model significantly outperformed individual baselines.

| Model Architecture | CV Accuracy | Kaggle Score (Public LB) |
| :--- | :--- | :--- |
| Random Forest | 85.9% | - |
| XGBoost | 86.8% | 0.779 |
| Logistic Regression | 86.9% | - |
| **Voting Ensemble (Final)** | **86.6%** | **0.794 (Rank ~600 / 12000)** |

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

- `01_Exploratory_Analysis.ipynb`: Detailed EDA, feature correlation heatmaps, and initial experiments.
- `02_Final_Pipeline.ipynb`: The clean, production-ready pipeline that generates the final submission.

## Project Structure

```plaintext
titanic-ensemble/
│
├── data/                       # Kaggle datasets
├── models/                     # Serialized models (.pkl)
├── src/                        # Python helper modules
│   ├── preprocessing.py        # Data transformers (impute, encode, feature logic)
│   └── evaluation.py           # Model tuning & visualization (GridSearch, ROC, Feature Importance)
├── submissions/                # Kaggle submission files (.csv)
│
├── 01_Exploratory_Analysis.ipynb   # Cleaned EDA & Feature investigation
├── 02_Final_Pipeline.ipynb         # End-to-end modeling & final solution
├── Titanic Machine Learning.ipynb  # (Draft/Sandbox) - Original messy notebook
├── environment.yml                 # Conda environment definition
└── README.md                       # Project documentation
```
