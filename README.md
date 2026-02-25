# Titanic Survival Prediction: Ensemble Learning Pipeline

An end-to-end Machine Learning pipeline that predicts passenger survival on the Titanic with **79.45% accuracy** (Top 5% kaggle participants leaderboard). 

This project moves beyond standard tutorials by implementing a **Voting Ensemble** (XGBoost, Random Forest, Logistic Regression) and engineering custom features to capture passenger group dynamics.

---

## Results

The ensemble model significantly outperformed individual baselines.

| Model Architecture | CV Accuracy | Kaggle Score (Public LB) |
| :--- | :--- | :--- |
| Random Forest | 85.9% | - |
| XGBoost | 86.8% | 0.779 |
| Logistic Regression | 86.9% | - |
| **Voting Ensemble (Final)** | **86.6%** | **0.794 (Rank ~600 / 12000)** |

## Technical Approach & Key Insights

### 1. Advanced Feature Engineering
* **Family Survival Context:** Rather than treating passengers as isolated data points, grouping them by Surname/Ticket captured the high correlation of family survival (families tended to live or die together). This context boosted model confidence by **~3%**.
* **Sociological Proxies:** The extracted `Title` feature served as a critical proxy for Age, Gender, and Social Class.
* **Model-Specific Transformations:** "Human-Logic" binning (e.g., Child vs. Adult) was implemented specifically to aid **Logistic Regression**, recognizing that while Tree models handle continuous splits well, linear models require explicit categorization to capture non-linear risks.
* **Redundancy Reduction:** Multicollinearity was addressed by dropping raw features like `SibSp`/`Parch` in favor of the derived `FamilySize`, significantly improving model stability.

### 2. Model Architecture
This project utilizes a **Voting Classifier** to combine three distinct models, balancing variance and bias:
* **XGBoost:** Captures complex, non-linear feature interactions.
* **Random Forest:** Provides stability and handles outliers via bagging.
* **Logistic Regression:** Acts as a robust linear baseline (regularized with L2/Ridge) to prevent overfitting on noise.

### 3. Critical Findings
* **The "Linear" Boundary:** While tree-based models provided strong baselines, the competitive performance of Logistic Regression post-engineering suggests that the decision boundary becomes largely linear once domain knowledge is explicitly encoded.
* **Combating Overfitting:** Given the small dataset size, strict constraints were necessary to force generalization. High regularization (`C`) was applied to linear models, while tree ensembles were restricted by limited depth and increased leaf samples.
* **The Iterative Cycle:** Success was achieved not through a single GridSearch, but through a continuous cycle of **Hypothesis $\rightarrow$ Feature Creation $\rightarrow$ Validation**, proving that statistical tuning must be paired with human intuition about the problem domain.

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
- `02_Final_Pipeline.ipynb`: **Recommended Read** The clean final pipeline & description that generates the final submission.
- `02_Final_Pipeline.ipynb` has ~80 second total runtime benchmarked on an Apple M4 chip.

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
