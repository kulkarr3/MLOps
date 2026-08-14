
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

XTRAIN_PATH = "tourism_project/data/Xtrain.csv"
XTEST_PATH = "tourism_project/data/Xtest.csv"
YTRAIN_PATH = "tourism_project/data/ytrain.csv"
YTEST_PATH = "tourism_project/data/ytest.csv"

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tourism_model.pkl"
)


# ---------------------------------------------------------
# 2. Load train and test splits
# ---------------------------------------------------------

print("Loading train and test data...")

X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)

y_train = pd.read_csv(YTRAIN_PATH).squeeze()
y_test = pd.read_csv(YTEST_PATH).squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")


# ---------------------------------------------------------
# 3. Define model
# ---------------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------------------------------
# 4. Define hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# ---------------------------------------------------------
# 5. MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment(
    "Tourism Package Prediction"
)


# ---------------------------------------------------------
# 6. Hyperparameter tuning + MLflow tracking
# ---------------------------------------------------------

with mlflow.start_run():

    print("\nStarting hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest CV F1 score: "
        f"{grid_search.best_score_:.4f}"
    )


    # -----------------------------------------------------
    # 7. Log best parameters
    # -----------------------------------------------------

    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metric(
        "best_cv_f1",
        grid_search.best_score_
    )


    # -----------------------------------------------------
    # 8. Evaluate best model
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # -----------------------------------------------------
    # 9. Log evaluation metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "test_precision",
        precision
    )

    mlflow.log_metric(
        "test_recall",
        recall
    )

    mlflow.log_metric(
        "test_f1",
        f1
    )


    # -----------------------------------------------------
    # 10. Print evaluation results
    # -----------------------------------------------------

    print("\n========== Model Evaluation ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # 11. Create MLflow model signature
    # -----------------------------------------------------

    input_example = X_train.head(1)

    prediction_example = best_model.predict(
        input_example
    )

    signature = infer_signature(
        input_example,
        prediction_example
    )


    # -----------------------------------------------------
    # 12. Log best model to MLflow
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="tourism_model",
        input_example=input_example,
        signature=signature
    )


    # -----------------------------------------------------
    # 13. Save best model to deployment folder
    # -----------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(
        "\nBest model saved successfully:"
    )

    print(MODEL_PATH)
