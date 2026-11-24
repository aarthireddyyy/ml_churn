import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"]
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_data(path):
    """Read CSV, fix types, return features X and target y."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df[NUM_COLS + CAT_COLS]
    return X, y


def add_features(X):
    """Engineered features: computed per row, so no leakage."""
    X = X.copy()
    X["n_services"] = (X[SERVICES] == "Yes").sum(axis=1)
    return X


def build_pipeline(model):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", num, NUM_COLS + ["n_services"]),
        ("cat", cat, CAT_COLS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", model),
    ])
