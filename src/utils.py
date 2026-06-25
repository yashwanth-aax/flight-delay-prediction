# src/utils.py

import joblib

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)


def evaluate_model(

        model,

        X_test,

        y_test

):

    pred = model.predict(

        X_test

    )


    prob = model.predict_proba(

        X_test

    )[:,1]


    metrics = {


        "Accuracy":

        accuracy_score(

            y_test,

            pred

        ),


        "Precision":

        precision_score(

            y_test,

            pred

        ),


        "Recall":

        recall_score(

            y_test,

            pred

        ),


        "F1":

        f1_score(

            y_test,

            pred

        ),


        "ROC_AUC":

        roc_auc_score(

            y_test,

            prob

        )

    }


    return metrics



def print_metrics(

        model_name,

        metrics

):

    print()

    print("="*50)

    print(model_name)

    print("="*50)


    for metric,value in metrics.items():

        print(

            f"{metric:<12}: {value:.4f}"

        )



def save_model(

        model,

        path

):

    joblib.dump(

        model,

        path

    )



def load_model(

        path

):

    return joblib.load(

        path

    )