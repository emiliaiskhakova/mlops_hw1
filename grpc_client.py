import grpc
import ml_service_pb2
import ml_service_pb2_grpc
import json


def run():
    # подключаемся к gRPC серверу на локальной машине
    channel = grpc.insecure_channel('localhost:50051')
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    # сначала проверяем, что сервер вообще работает и отвечает
    print("Testing HealthCheck...")
    health_response = stub.HealthCheck(ml_service_pb2.HealthRequest())
    print(f"Health: {health_response.status}, Time: {health_response.timestamp}")

    #какие модели доступны для обучения
    print("\nTesting ListModels...")
    models_response = stub.ListModels(ml_service_pb2.ListModelsRequest())
    print(f"Available models: {models_response.available_models}")

    # параметры по умолчанию
    print("\nTesting GetModelParams...")
    params_response = stub.GetModelParams(ml_service_pb2.GetModelParamsRequest(model_name="LogisticRegression"))
    print(f"Model: {params_response.model_name}")
    print(f"Params: {dict(params_response.default_hyperparameters)}")

    # обучаем модель
    print("\nTesting TrainModel...")
    train_data = {
        "X": [
            [1.0, 2.0], [2.0, 1.5], [1.5, 1.8], [2.2, 1.9],
            [1.2, 2.3], [2.5, 1.2], [1.8, 2.1], [2.1, 1.7],
            [1.4, 2.4], [2.3, 1.3], [1.7, 2.2], [2.4, 1.4]
        ],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }

    # формируем запрос на обучение с выбранной моделью и параметрами
    train_request = ml_service_pb2.TrainRequest(
        model_name="LogisticRegression",
        hyperparams={"C": "0.5"},
        data_json=json.dumps(train_data)
    )

    try:
        # отправляем запрос на сервер и ждем, пока модель обучится
        train_response = stub.TrainModel(train_request)
        print(f"Trained model ID: {train_response.model_id}")
        # Форматируем метрики с 4 знаками после запятой
        formatted_metrics = {k: round(v, 4) for k, v in train_response.metrics.items()}
        print(f"Metrics: {formatted_metrics}")
        
        model_id = train_response.model_id

        print("\nTesting Predict...")
        
        # готовим новые данные
        predict_data = [
            [1.2, 2.1],
            [1.8, 1.6],
            [2.0, 1.9]
        ]

        predict_request = ml_service_pb2.PredictRequest(
            model_id=model_id,
            data_json=json.dumps(predict_data)
        )

        # получаем предсказания
        predict_response = stub.Predict(predict_request)
        print(f"Predictions: {predict_response.predictions}")

        # смотрим список всех обученных моделей
        print("\nTesting ListTrainedModels...")
        trained_response = stub.ListTrainedModels(ml_service_pb2.ListTrainedModelsRequest())
        for model in trained_response.models:
            print(f"Model: {model.model_name}, ID: {model.model_id}")

    except grpc.RpcError as e:
        # обрабатываем ошибки
        print(f"gRPC Error: {e.details()}")
        print(f"Error code: {e.code()}")

    except Exception as e:
        # другие ошибки
        print(f"Error: {e}")


if __name__ == '__main__':
    run()

