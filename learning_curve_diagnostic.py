import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve

warnings.filterwarnings("ignore")


def main():
    # Load dataset
    df = pd.read_csv("data/telecom_churn.csv")

    # Define target and remove non-useful ID column if it exists
    target_col = "churned"
    drop_cols = [target_col]

    if "customer_id" in df.columns:
        drop_cols.append("customer_id")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Show basic dataset info
    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))

    # Detect numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\nNumeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("nums", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use at least 5 training sizes
    train_sizes = np.linspace(0.1, 1.0, 6)

    # Generate learning curves
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    # Mean and std across folds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Print summary
    print("\nTrain sizes:", train_sizes_abs)
    print("Train mean:", train_mean)
    print("Validation mean:", val_mean)
    print("Train std:", train_std)
    print("Validation std:", val_std)

    # Save plot
    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_mean, marker="o", label="Training F1")
    plt.plot(train_sizes_abs, val_mean, marker="o", label="Validation F1")

    plt.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2
    )
    plt.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2
    )

    plt.title("Learning Curve for Logistic Regression on Telecom Churn")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/learning_curve.png")
    plt.close()

    print("\nPlot saved to output/learning_curve.png")
    print("Scoring metric used: F1 because the dataset is imbalanced.")


if __name__ == "__main__":
    main()