import gradio as gr
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

REST_API_URL = "http://127.0.0.1:8000"


def health_check():
    try:
        response = requests.get(f"{REST_API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            # Безопасное извлечение данных
            status = data.get('status', 'unknown')
            timestamp = data.get('timestamp', 'unknown')
            service = data.get('service', 'ML Training API')
            return f" Status: {status}\n Timestamp: {timestamp}\n Service: {service}"
        else:
            return f"HTTP Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Cannot connect to service: {str(e)}"


def list_models():
    #Список доступных моделей
    try:
        response = requests.get(f"{REST_API_URL}/models")
        if response.status_code == 200:
            models = response.json()["available_models"]
            return "\n".join([f"• {model}" for model in models])
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_model_params(model_name):
    # Получение параметров модели
    try:
        response = requests.get(f"{REST_API_URL}/model_params/{model_name}")
        if response.status_code == 200:
            params = response.json()["default_hyperparameters"]
            return json.dumps(params, indent=2)
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def train_model(model_name, hyperparams_json, data_json):
    #Обучение модели
    try:
        # Парсим гиперпараметры
        hyperparams = json.loads(hyperparams_json) if hyperparams_json.strip() else {}
        # Парсим данные
        data = json.loads(data_json)

        request_data = {
            "model_name": model_name,
            "hyperparams": hyperparams,
            "data": data
        }

        response = requests.post(f"{REST_API_URL}/train", json=request_data)
        if response.status_code == 200:
            result = response.json()
            if 'metrics' in result:
                for metric_name, metric_value in result['metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        result['metrics'][metric_name] = round(metric_value, 4)
            return json.dumps(result, indent=2)
        else:
            return f"Error: {response.text}"

    except json.JSONDecodeError as e:
        return f"JSON Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def list_trained_models():
    #Список обученных моделей
    try:
        response = requests.get(f"{REST_API_URL}/trained_models")
        if response.status_code == 200:
            models = response.json()["trained_models"]
            if not models:
                return "No trained models available"

            result = []
            for model in models:
                result.append(f"Model: {model['model_name']}")
                result.append(f"   ID: {model['model_id']}")
                result.append(f"   Trained: {model['training_time']}")
                formatted_metrics = {}
                result.append(f"    Metrics: {json.dumps(formatted_metrics, indent=2)}")
                result.append("")

            return "\n".join(result)
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def predict_model(model_id, data_json):
    #предсказание модели
    try:
        data = json.loads(data_json)
        request_data = {
            "model_id": model_id,
            "data": data
        }

        response = requests.post(f"{REST_API_URL}/predict", json=request_data)
        if response.status_code == 200:
            result = response.json()

            # Создаем визуализацию
            predictions = result['predictions']
            fig = go.Figure(data=[go.Bar(x=list(range(len(predictions))), y=predictions)])
            fig.update_layout(
                title=f"Predictions from {result['model_name']}",
                xaxis_title="Sample Index",
                yaxis_title="Prediction"
            )

            return json.dumps(result, indent=2), fig
        else:
            return f"Error: {response.text}", None

    except Exception as e:
        return f"Error: {str(e)}", None


def delete_model(model_id):
    # удаление модели
    try:
        response = requests.delete(f"{REST_API_URL}/model/{model_id}")
        if response.status_code == 200:
            return f"{response.json()['message']}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# данные по умолчанию
DEFAULT_TRAINING_DATA = {
    "features": [
        {"f1": 1.2, "f2": 2.3, "f3": 0.8},
        {"f1": 2.1, "f2": 1.5, "f3": 1.2},
        {"f1": 1.8, "f2": 1.9, "f3": 0.9},
        {"f1": 2.4, "f2": 2.1, "f3": 1.5},
        {"f1": 1.5, "f2": 2.5, "f3": 0.7},
        {"f1": 2.7, "f2": 1.3, "f3": 1.8},
        {"f1": 1.9, "f2": 2.2, "f3": 1.1},
        {"f1": 2.2, "f2": 1.7, "f3": 1.4},
        {"f1": 1.6, "f2": 2.4, "f3": 0.6},
        {"f1": 2.8, "f2": 1.4, "f3": 1.7},
        {"f1": 1.4, "f2": 2.6, "f3": 0.5},
        {"f1": 2.5, "f2": 1.6, "f3": 1.6},
        {"f1": 2.0, "f2": 2.0, "f3": 1.0},
        {"f1": 1.7, "f2": 1.8, "f3": 1.2},
        {"f1": 2.3, "f2": 1.9, "f3": 0.9}
    ],
    "labels": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

DEFAULT_PREDICT_DATA = [
    {"f1": 1.5, "f2": 2.0, "f3": 0.9},
    {"f1": 2.0, "f2": 1.7, "f3": 1.3},
    {"f1": 1.8, "f2": 2.3, "f3": 1.1},
    {"f1": 2.2, "f2": 1.9, "f3": 0.9},
    {"f1": 1.9, "f2": 1.8, "f3": 1.2}
]

# Создаем интерфейс Gradio
with gr.Blocks(title="ML Model Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#ML Model Training Dashboard")
    gr.Markdown("Interactive dashboard for training and managing ml models")

    with gr.Tab("Health Check"):
        gr.Markdown("### Service Status")
        health_btn = gr.Button("Check Service Health", variant="primary")
        health_output = gr.Textbox(label="Status", lines=4, interactive=False)
        health_btn.click(health_check, outputs=health_output)

    with gr.Tab("Available Models"):
        gr.Markdown("### Available Model Classes")
        models_btn = gr.Button("List Available Models", variant="primary")
        models_output = gr.Textbox(label="Models", lines=6, interactive=False)
        models_btn.click(list_models, outputs=models_output)

        gr.Markdown("### Model Parameters")
        model_selector = gr.Dropdown(
            choices=["LinearRegression", "LogisticRegression", "RandomForestRegressor", "RandomForestClassifier"],
            label="Select Model",
            value="LogisticRegression"
        )
        params_btn = gr.Button("Get Default Parameters")
        params_output = gr.Textbox(label="Default Hyperparameters", lines=6)
        params_btn.click(get_model_params, inputs=model_selector, outputs=params_output)

    with gr.Tab("Train Model"):
        gr.Markdown("### Train New Model")

        with gr.Row():
            with gr.Column():
                train_model_name = gr.Dropdown(
                    choices=["LinearRegression", "LogisticRegression", "RandomForestRegressor",
                             "RandomForestClassifier"],
                    label="Model Type",
                    value="LogisticRegression"
                )
                train_hyperparams = gr.Textbox(
                    label="Hyperparameters (JSON)",
                    value='{"C": 0.5, "max_iter": 1000}',
                    lines=3,
                    placeholder='{"param1": value1, "param2": value2}'
                )
            with gr.Column():
                train_data = gr.Textbox(
                    label="Training Data (JSON)",
                    value=json.dumps(DEFAULT_TRAINING_DATA, indent=2),
                    lines=10
                )

        train_btn = gr.Button("Train Model", variant="primary")
        train_output = gr.Textbox(label="Training Result", lines=8)

        train_btn.click(
            train_model,
            inputs=[train_model_name, train_hyperparams, train_data],
            outputs=train_output
        )

    with gr.Tab("Trained Models"):
        gr.Markdown("### Manage Trained Models")

        trained_btn = gr.Button("Refresh List", variant="primary")
        trained_output = gr.Textbox(label="Trained Models", lines=10, interactive=False)
        trained_btn.click(list_trained_models, outputs=trained_output)

        gr.Markdown("### Delete Model")
        delete_model_id = gr.Textbox(label="Model ID to Delete")
        delete_btn = gr.Button("Delete Model", variant="stop")
        delete_output = gr.Textbox(label="Delete Result", interactive=False)
        delete_btn.click(delete_model, inputs=delete_model_id, outputs=delete_output)

    with gr.Tab("Predict"):
        gr.Markdown("### Make Predictions")

        with gr.Row():
            with gr.Column():
                predict_model_id = gr.Textbox(
                    label="Model ID",
                    placeholder="Enter model ID from trained models list"
                )
                predict_data = gr.Textbox(
                    label="Data for Prediction (JSON)",
                    value=json.dumps(DEFAULT_PREDICT_DATA, indent=2),
                    lines=6
                )
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                predict_output = gr.Textbox(label="Prediction Result", lines=6)
                plot_output = gr.Plot(label="Predictions Visualization")

        predict_btn.click(
            predict_model,
            inputs=[predict_model_id, predict_data],
            outputs=[predict_output, plot_output]
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## ML Training API Dashboard

        This dashboard provides a user-friendly interface for:
        - Training machine learning models
        - Managing trained models
        - Making predictions
        - Monitoring service health

        ### Supported Models:
        - **LinearRegression** (Lasso)
        - **LogisticRegression**
        - **RandomForestRegressor**
        - **RandomForestClassifier**

        ### API Endpoints:
        - REST API: http://localhost:8000
        - Documentation: http://localhost:8000/docs
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )