from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import logging
import joblib
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Модели и дефолтные гиперпараметры
dict_model = {
    'LinearRegression': Lasso,
    'LogisticRegression': LogisticRegression,
    'RandomForestRegressor': RandomForestRegressor,
    'RandomForestClassifier': RandomForestClassifier
}

defolt_param_dict = {
    "LogisticRegression": {"C": 1.0, "max_iter": 1000, "penalty": "l2", "random_state": 42},
    "RandomForestClassifier": {"n_estimators": 100, "max_depth": 1000, "min_samples_split": 2, "random_state": 42},
    "LinearRegression": {"alpha": 1.0, "max_iter": 1000, "selection": "random", "random_state": 42},
    "RandomForestRegressor": {"n_estimators": 100, "max_depth": 1000, "min_samples_split": 2, "random_state": 42}
}

# Хранилище обученных моделей
trained_models = {}


def prepare_date(data, y):
    """Извлечение данных, разные форматы"""
    if isinstance(data, dict):
        if "X" in data and "y" in data:
            df = pd.DataFrame(data["X"])
            y = pd.Series(data["y"])
        elif "features" in data and "labels" in data:
            features = data["features"]
            if isinstance(features[0], dict):
                df = pd.DataFrame(features)
            else:
                df = pd.DataFrame(features)
            y = pd.Series(data["labels"])
        else:
            raise ValueError("Неизвестный формат JSON")
    elif isinstance(data, str):
        df = pd.read_csv(io.StringIO(data))
        y = df.iloc[:, -1]
        df = df.iloc[:, :-1]
    elif isinstance(data, bytes):
        df = pd.read_csv(io.BytesIO(data))
        y = df.iloc[:, -1]
        df = df.iloc[:, :-1]
    else:
        raise ValueError("Unsupported data format")

    return df, y


def trainer(df, y, model_name, hyperparams: dict):
    """Обучение модели"""
    df, y= prepare_date(df, y)

    if model_name not in dict_model:
        raise ValueError(f"Model {model_name} is not available")
 
    if model_name in ['LinearRegression', 'RandomForestRegressor']:
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=None)
    else:
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

    ModelClass = dict_model[model_name]
    params = {**defolt_param_dict.get(model_name, {}), **hyperparams}
    model = ModelClass(**params)
    model.fit(df_train, y_train)

    y_pred = model.predict(df_test)
    if model_name in ['LinearRegression', 'RandomForestRegressor']:
        y_pred_proba = None
    else: 
        y_pred_proba = model.predict_proba(df_test)[:, 1]
    return model, y_pred, y_test, y_pred_proba


def calculate_metrics(model_name, y_test, y_pred, y_pred_proba):
    """Возвращает предсказания и метрики"""
    y_pred_list = y_pred.tolist()
    if model_name in ['LinearRegression', 'RandomForestRegressor']:
        return {
            "prediction": y_pred_list, 
            "r2_score": r2_score(y_test, y_pred),
            "mean_squared_error": mean_squared_error(y_test, y_pred)
        }
    else:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = None

        return {
            "prediction": y_pred_list, 
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "roc_auc": roc_auc
        }


# FastAPI часть
# -----------------------------------

app = FastAPI(title="ML Training API", description="API для обучения ML-модель с возможностью настройки гиперпараметров", version="1.0")

class TrainRequest(BaseModel):
    model_name: str
    hyperparams: Dict[str, Any] = {}
    data: Dict[str, Any]


class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, float]]


class RetrainRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка статуса сервиса"""
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models", tags=["Models"])
async def list_models():
    """Возвращает список доступных классов моделей"""
    logger.info("List models requested")
    return {"available_models": list(dict_model.keys())}


@app.get("/model_params/{model_name}", tags=["Models"])
async def get_model_params(model_name: str):
    """Возвращает параметры по умолчанию для указанной модели"""
    logger.info(f"Model params requested for {model_name}")
    if model_name not in defolt_param_dict:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_name": model_name, "default_hyperparameters": defolt_param_dict[model_name]}


@app.post("/train", tags=["Training"])
async def train_model(request: TrainRequest):
    """Обучает новую модель"""
    logger.info(f"Training model: {request.model_name}")

    try:
        model, y_pred, y_test, y_pred_proba = trainer(
            request.data,
            None,  
            request.model_name,
            request.hyperparams
        )

        metrics_result = calculate_metrics(request.model_name, y_test, y_pred, y_pred_proba)

        # Сохраняем модель
        model_id = str(uuid.uuid4())
        trained_models[model_id] = {
            "model": model,
            "model_name": request.model_name,
            "hyperparams": request.hyperparams,
            "training_time": datetime.now(),
            "metrics": metrics_result
        }

        logger.info(f"Model trained successfully: {model_id}")

        return {
            "model_id": model_id,
            "model_name": request.model_name,
            "metrics": metrics_result,
            "message": "Model trained successfully"
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/trained_models", tags=["Models"])
async def list_trained_models():
    """Возвращает список всех обученных моделей"""
    logger.info("List trained models requested")
    models_list = []
    for model_id, model_info in trained_models.items():
        models_list.append({
            "model_id": model_id,
            "model_name": model_info["model_name"],
            "hyperparams": model_info["hyperparams"],
            "training_time": model_info["training_time"],
            "metrics": model_info["metrics"]
        })
    return {"trained_models": models_list}


@app.post("/predict", tags=["Prediction"])
async def predict(request: PredictRequest):
    """Возвращает предсказание конкретной модели"""
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model_info = trained_models[request.model_id]
        model = model_info["model"]

        # Преобразуем данные для предсказания
        df = pd.DataFrame(request.data)
        predictions = model.predict(df)

        logger.info(f"Prediction completed for model: {request.model_id}")

        return {
            "model_id": request.model_id,
            "model_name": model_info["model_name"],
            "predictions": predictions.tolist()
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/retrain/{model_id}", tags=["Training"])
async def retrain_model(model_id: str, request: RetrainRequest):
    """Переобучает существующую модель"""
    logger.info(f"Retraining model: {model_id}")

    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model_info = trained_models[model_id]
        model_name = model_info["model_name"]

        model, y_pred, y_test, y_pred_proba = trainer(
            request.data,
            None,
            model_name,
            model_info["hyperparams"]
        )

        metrics_result = calculate_metrics(model_name, y_test, y_pred, y_pred_proba)

        # Обновляем модель
        trained_models[model_id] = {
            "model": model,
            "model_name": model_name,
            "hyperparams": model_info["hyperparams"],
            "training_time": datetime.now(),
            "metrics": metrics_result
        }

        logger.info(f"Model retrained successfully: {model_id}")

        return {
            "model_id": model_id,
            "model_name": model_name,
            "hyperparams": model_info["hyperparams"],  
            "training_time": datetime.now(),
            "metrics": metrics_result,
            "message": "Model retrained successfully"
        }

    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/model/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """Удаляет обученную модель"""
    logger.info(f"Deleting model: {model_id}")

    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    del trained_models[model_id]
    logger.info(f"Model deleted: {model_id}")

    return {"message": f"Model {model_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
