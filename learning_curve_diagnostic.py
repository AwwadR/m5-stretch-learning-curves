import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("data/telecom_churn.csv")

target_col = "churned"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
numeric_cols = X.select_dtypes(include=["int64", "Float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("nums", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# CV and training sizes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes = np.linspace(0.1, 1.0, 6)

# Learning curve
train_sizes_abs, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

# Means and std
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Print summary
print("Train sizes:", train_sizes_abs)
print("Train mean:", train_mean)
print("Validation mean:", val_mean)
print("Train std:", train_std)
print("Validation std:", val_std)

# Plot
os.makedirs("output", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes_abs, train_mean, marker="o", label="Training F1")
plt.plot(train_sizes_abs, val_mean, marker="o", label="Validation F1")

plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.title("Learning Curve: Logistic Regression (F1 Score)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/learning_curve.png")
plt.show()