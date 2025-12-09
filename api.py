# print("Hello depuis Docker !")

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello depuis FastAPI et Docker et fastapi"}


@app.get("/health")
def health():
    return {"status": "OK"}


class PredictInput(BaseModel):
    age: int
    revenu_estime_mois: float

@app.post("/predict")
def predict(payload: PredictInput):
    import mlflow
    from modules.preprocess import preprocessing, split
    from modules.evaluate import evaluate_performance
    from modules.print_draw import print_data, draw_loss
    from models.models import create_nn_model, train_model, model_predict
    import pandas as pd
    import joblib
    from os.path import join as join

    # mlflow.set_tracking_uri('http://localhost:5000')

    model_name = "model_2025_12"
    model_filename = f"{model_name}.pkl"

    # Chargement des datasets
    df_new = pd.read_csv(join('data','df_new.csv'))
    # df_input = pd.DataFrame([payload.dict()])  # une ligne avec age / salary
    # print(df_input)

    # Charger le préprocesseur
    preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))
    # charger le modèle
    loaded_model = joblib.load(join('models', model_filename))
    # loaded_model = mlflow.sklearn.load_model(f'runs:/{run_id}/{model_name}')


    # preprocesser les data
    X, y, _ = preprocessing(df_new)

    # split data in train and test dataset
    X_train, X_test, y_train, y_test = split(X, y)


    #%% predire sur les valeurs de train
    y_pred = model_predict(loaded_model, X_train)
    # print(y_pred)

    # mesurer les performances MSE, MAE et R²
    perf = evaluate_performance(y_train, y_pred)
    #print_data(perf)

    data = {
        "mse": perf["MSE"],
        "mae": perf["MAE"],
        "r2": perf["R²"]
    }
    return data


@app.post("/retrain")
def retrain():
    return {"message": "TODO"}