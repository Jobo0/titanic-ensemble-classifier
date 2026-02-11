import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

# models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# XGBoost
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labes vs y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2), "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:0.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models: a dict of different Scikit-Learn machine learning models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    y_train: training labels
    y_test: test labels
    """
    # set random seed
    np.random.seed(12)
    # make a dictionary to keep model scores
    model_scores = {}
    # loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)

    return model_scores

def cross_validated_scores(model, X, y, cv=5, scoring_list=["accuracy", "precision", "recall", "f1"]):
    """
    Runs cross-validation for multiple scoring metrics and returns a dictionary of mean scores.
    """
    results = {}
    
    for score_name in scoring_list:
        # Run CV
        scores = cross_val_score(model, X, y, cv=cv, scoring=score_name)
        
        # Store the mean score
        results[score_name] = np.mean(scores)
        
    return results

def train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test, X=None, y=None, feature_engineering=None, param_grid=None):
    """
    1. Runs GridSearch on X_train/y_train
    2. Evaluates the best model on X_test/y_test
    3. Plots metrics
    """
    # 1. Dynamic Pipeline Construction
    pipeline_steps = []
    
    if feature_engineering:
        pipeline_steps.append(('feat_eng', feature_engineering))
    
    pipeline_steps.append(('preprocessor', preprocessor))
    pipeline_steps.append(('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
    
    pipeline = Pipeline(steps=pipeline_steps)

    # 2. Define Default Hyperparameters
    if param_grid is None:
        param_grid = {
            'model__n_estimators': [100, 500],
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8]
        }

    # 3. GridSearch (Splits Training Data Internally for Validation)
    print(f"🔍 Tuning hyperparameters...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Best Params: {grid_search.best_params_}")
    
    # 4. Predict on the CONSTANT Test Set
    y_preds = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 5. Evaluation Metrics
    print("\n" + "="*40)
    print("📊 CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_test, y_preds))

    # 6. Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ConfusionMatrixDisplay.from_predictions(y_test, y_preds, cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")

    # 7. Cross Validation
    print("\n" + "="*40)
    print("CROSS-VALIDATED SCORES ON X AND Y")
    print("="*40)
    if X is None or y is None:
        print("No Cross Validation Set Provided")
    else:
        cv_scores = cross_validated_scores(best_model, X, y, 5)
        print(cv_scores)
    
    # Feature Importance Logic
    try:
        # Get feature names from the preprocessor step
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['model'].feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        feat_imp.tail(15).plot(kind='barh', ax=axes[2], color='teal')
        axes[2].set_title("Top 15 Feature Importances")
    except Exception as e:
        print(f"⚠️ Could not plot feature importance: {e}")
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return best_model