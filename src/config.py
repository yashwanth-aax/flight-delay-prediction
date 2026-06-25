from pathlib import Path

# ==============================
# Project Paths
# ==============================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "flights.csv"

MODELS_DIR = ROOT / "models"

# ==============================
# Data Settings
# ==============================

SAMPLE_SIZE = 300000

RANDOM_STATE = 42

# ==============================
# Features
# ==============================

FEATURES = [

    "MONTH",

    "AIRLINE",

    "ORIGIN_AIRPORT",

    "DESTINATION_AIRPORT",

    "SCHEDULED_DEPARTURE",

    "SCHEDULED_TIME",

    "DISTANCE",

    "DEPARTURE_DELAY",

    "TAXI_OUT",

    "DEPARTURE_HOUR",

    "IS_MORNING"

]

TARGET = "Delayed"