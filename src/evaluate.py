# src/evaluate.py

from pathlib import Path

import joblib

from src.preprocess import load_data

from utils import (

    evaluate_model,

    print_metrics

)

from config import MODELS_DIR



def evaluate_all():

    X_train,X_test,y_train,y_test,_ = load_data()


    models = {

        "Logistic Regression":

        MODELS_DIR/"logistic_regression.pkl",


        "Random Forest":

        MODELS_DIR/"random_forest.pkl",


        "XGBoost":

        MODELS_DIR/"xgboost.pkl"

    }


    for name,path in models.items():

        print()

        print("="*50)

        print(name)

        print("="*50)


        model = joblib.load(path)


        metrics = evaluate_model(

            model,

            X_test,

            y_test

        )


        print_metrics(

            name,

            metrics

        )



if __name__=="__main__":

    evaluate_all()