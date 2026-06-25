# src/train.py
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import load_data
from src.utils import (
    evaluate_model,
    print_metrics,
    save_model
)
from src.config import MODELS_DIR


def train_all():

    print("Loading processed data...\n")

    X_train, X_test, y_train, y_test, encoders = load_data()

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save encoders
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos_weight = neg / pos

    print(f"\nNegative samples : {neg}")
    print(f"Positive samples : {pos}")
    print(f"scale_pos_weight : {scale_pos_weight:.2f}\n")

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )

    }

    for name, model in models.items():

        print("\n")
        print("=" * 60)
        print(name)
        print("=" * 60)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        print_metrics(name, metrics)

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        save_model(
            model,
            MODELS_DIR / filename
        )

        print(f"\nSaved -> {filename}")

        # -----------------------------
        # Feature Importance (XGBoost)
        # -----------------------------

        if name == "XGBoost":

            importance = pd.DataFrame({

                "Feature": X_train.columns,

                "Importance": model.feature_importances_

            })

            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )

            print("\n")
            print("=" * 60)
            print("XGBoost Feature Importance")
            print("=" * 60)

            print(importance.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_all()