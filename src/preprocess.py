# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import *


def load_data():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        dtype={
            "AIRLINE": str,
            "ORIGIN_AIRPORT": str,
            "DESTINATION_AIRPORT": str
        }
    )

    print("Sampling dataset...")

    df = df.sample(
        SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print("Selecting required columns...")

    df = df[
        [
            "MONTH",
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "DISTANCE",
            "DEPARTURE_DELAY",
            "TAXI_OUT",
            "ARRIVAL_DELAY"
        ]
    ]

    print("Dropping missing values...")

    df.dropna(inplace=True)

    print("Creating target variable...")

    df[TARGET] = (
        df["ARRIVAL_DELAY"] > 15
    ).astype(int)

    # ---------------------------------
    # Feature Engineering
    # ---------------------------------

    print("Creating engineered features...")

    df["DEPARTURE_HOUR"] = (
        df["SCHEDULED_DEPARTURE"] // 100
    )

    df["IS_MORNING"] = (
        (
            (df["DEPARTURE_HOUR"] >= 5)
            &
            (df["DEPARTURE_HOUR"] < 12)
        )
    ).astype(int)

    # ---------------------------------
    # Encode categorical variables
    # ---------------------------------

    categorical = [

        "AIRLINE",

        "ORIGIN_AIRPORT",

        "DESTINATION_AIRPORT"

    ]

    encoders = {}

    print("Encoding categorical columns...")

    for col in categorical:

        df[col] = df[col].astype(str)

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col]
        )

        encoders[col] = encoder

    # ---------------------------------
    # Features and Target
    # ---------------------------------

    X = df[FEATURES]

    y = df[TARGET]

    print("Splitting train and test data...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y

    )

    print()

    print("Train Shape :", X_train.shape)

    print("Test Shape  :", X_test.shape)

    print()

    print("Class Distribution")

    print(y_train.value_counts())

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        encoders

    )


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, encoders = load_data()

    print("\nData Loaded Successfully!\n")

    print("Features Used:")

    print(FEATURES)

    print()

    print("Training Samples :", X_train.shape)

    print("Testing Samples  :", X_test.shape)

    print()

    print(X_train.head())