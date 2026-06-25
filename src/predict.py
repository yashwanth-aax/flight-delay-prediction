# src/predict.py

import joblib

import pandas as pd

from src.config import MODELS_DIR



model = joblib.load(

    MODELS_DIR/"xgboost.pkl"

)


encoders = joblib.load(

    MODELS_DIR/"encoders.pkl"

)



def predict(

    month,

    day,

    day_of_week,

    airline,

    origin,

    destination,

    departure,

    scheduled_time,

    distance

):

    airline = encoders[

        "AIRLINE"

    ].transform(

        [airline]

    )[0]


    origin = encoders[

        "ORIGIN_AIRPORT"

    ].transform(

        [origin]

    )[0]


    destination = encoders[

        "DESTINATION_AIRPORT"

    ].transform(

        [destination]

    )[0]


    X = pd.DataFrame({

        "MONTH":[month],

        "DAY":[day],

        "DAY_OF_WEEK":[day_of_week],

        "AIRLINE":[airline],

        "ORIGIN_AIRPORT":[origin],

        "DESTINATION_AIRPORT":[destination],

        "SCHEDULED_DEPARTURE":[departure],

        "SCHEDULED_TIME":[scheduled_time],

        "DISTANCE":[distance]

    })


    pred = model.predict(

        X

    )[0]


    prob = model.predict_proba(

        X

    )[0][1]


    return pred,prob