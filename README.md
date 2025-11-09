# mlops
курс ФТиАД "Запуск ML моделей в промышленной среде"

## Состав команды
Беленок Анна, Исхакова Эмилия, Кияшко Полина

В работе присутствуют следующие файлы: main.py, dashboard_gradio.py, grpc_client.py, grpc_server.py, ml_service.proto, ml_service_pb2.py, ml_service_grpc.py, poetry.lock, pyproject.toml
## Основная часть
Для запуска основной части следует установить poetry и запустить
```
pip install poetry
```
```
install poetry
```
и запуск main.py
```
python main.py
```

Далее доступен переход на http://localhost:8000 где можно проверить статус сервиса, получить список используемых моделей: "LinearRegression", "LogisticRegression",  "RandomForestRegressor", "RandomForestClassifier", обучить ML-модель с настройкой гиперпараметров, вернуть предсказание конкретной модели.

### Дашборд gradio 
Для работы с дашбордом gradio следует в отдельном терминале запустить dashboard_gradio.py через 
```
python dashboard_gradio.py
```
Далее доступен переход на http://localhost:7860/ где доступны те же функции

# Инструкция для запуска gRPC ML Service (3-4 п)

## Описание
В рамках выполнения этого этого задания мы использовали **gRPC-сервер (`grpc_server.py`)** и **клиент (`grace_client.py`)** для обучения и предсказаний ML-моделей. Также спользуются функции из `main.py` и словарь `trained_models`.

## Установка.
## Для начала необходимо установить библиотеку для grpc:
```bash
pip install grpcio grpcio-tools
```

## Запуск
### Для начала запускаем grpc_server:
```bash
python3 grpc_server.py
```
### Должны получить следующий вывод:
```bash
gRPC server running on port 50051
```

### Затем вызываем клиент для тестирования (в новом терминале):
```bash
python3 grpc_client.py
```
### При отсутствии ошибок получаем следующие результаты тестирования:
```bash
Testing HealthCheck...
Health: healthy, Time: 2025-11-08T19:06:19.717413

Testing ListModels...
Available models: ['LinearRegression', 'LogisticRegression', 'RandomForestRegressor', 'RandomForestClassifier']

Testing GetModelParams...
Model: LogisticRegression
Params: {'penalty': 'l2', 'max_iter': '1000', 'C': '1.0'}

Testing TrainModel...
Trained model ID: 9dacc703-3159-4f9d-a81f-6721eccea55f
Metrics: {'f1_score': 0.6667, 'roc_auc': 0.75, 'accuracy': 0.6667}

Testing Predict...
Predictions: [0.0, 1.0, 1.0]

Testing ListTrainedModels...
Model: LogisticRegression, ID: 9dacc703-3159-4f9d-a81f-6721eccea55f
```
* Последовательно вызываются: HealthCheck, ListModels, GetModelParams, TrainModel, Predict, ListTrainedModels.
* gRPC файлы (ml_service_pb2.py, _grpc.py) описывают методы.
* Сервер принимает запросы, вызывает функции из main.py, сохраняет модели в trained_models (порт для подключения испоользуется 50051)
* Затем клиент посылает JSON с данными и гиперпараметры.
* Результаты обучения возвращаются с model_id для последующих операций.
* Ошибки возвращаются через context.set_code() и set_details()
