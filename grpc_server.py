import grpc
from concurrent import futures
import logging
import json
import ml_service_pb2
import ml_service_pb2_grpc
from main import dict_model, defolt_param_dict, trainer, calculate_metrics
import uuid
from datetime import datetime

# используем то же хранилище, что и в REST API, чтобы модели были доступны из обоих интерфейсов
from main import trained_models


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    
    def HealthCheck(self, request, context):
        # просто проверяем, что сервис жив и возвращаем текущее время
        return ml_service_pb2.HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat()
        )

    def ListModels(self, request, context):
        # показываем клиенту, какие модели вообще доступны для обучения
        return ml_service_pb2.ListModelsResponse(
            available_models=list(dict_model.keys())
        )

    def GetModelParams(self, request, context):
        # отдаем параметры по умолчанию для выбранной модели
        if request.model_name not in defolt_param_dict:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return ml_service_pb2.GetModelParamsResponse()

        # конвертируем параметры в строки, так как gRPC плохо работает с числами напрямую
        hyperparams = {k: str(v) for k, v in defolt_param_dict[request.model_name].items()}
        return ml_service_pb2.GetModelParamsResponse(
            model_name=request.model_name,
            default_hyperparameters=hyperparams
        )

    def TrainModel(self, request, context):
        try:
            # парсим данные и гиперпараметры из запроса
            data = json.loads(request.data_json)
            hyperparams = {k: float(v) if '.' in v else int(v) for k, v in request.hyperparams.items()}

            model, y_pred, y_test = trainer(
                data, None, request.model_name, hyperparams
            )

            # считаем метрики качества модели
            metrics_result = calculate_metrics(request.model_name, y_test, y_pred)

            # создаем уникальный ID для модели и сохраняем ее
            model_id = str(uuid.uuid4())
            trained_models[model_id] = {
                "model": model,
                "model_name": request.model_name,
                "hyperparams": hyperparams,
                "training_time": datetime.now(),
                "metrics": metrics_result
            }

            # конвертируем метрики в формат, понятный gRPC
            metrics_proto = {k: float(v) for k, v in metrics_result.items()}

            return ml_service_pb2.TrainResponse(
                model_id=model_id,
                model_name=request.model_name,
                metrics=metrics_proto,
                message="Model trained successfully"
            )

        except Exception as e:
            #сообщаем об ошибке клиенту
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.TrainResponse()

    def ListTrainedModels(self, request, context):
        # показываем все модели, которые уже были обучены
        models = []
        for model_id, model_info in trained_models.items():
            metrics_proto = {k: float(v) for k, v in model_info["metrics"].items()}
            models.append(ml_service_pb2.TrainedModelInfo(
                model_id=model_id,
                model_name=model_info["model_name"],
                training_time=model_info["training_time"].isoformat(),
                metrics=metrics_proto
            ))
        return ml_service_pb2.ListTrainedModelsResponse(models=models)

    def Predict(self, request, context):
        # используем обученную модель для предсказаний
        if request.model_id not in trained_models:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return ml_service_pb2.PredictResponse()

        try:
            model_info = trained_models[request.model_id]
            model = model_info["model"]
            data = json.loads(request.data_json)

            # преобразуем данные в формат, понятный модели
            import pandas as pd
            df = pd.DataFrame(data)
            predictions = model.predict(df)

            return ml_service_pb2.PredictResponse(
                model_id=request.model_id,
                model_name=model_info["model_name"],
                predictions=predictions.tolist()
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse()

    def RetrainModel(self, request, context):
        # переобучаем существующую модель на новых данных
        if request.model_id not in trained_models:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return ml_service_pb2.RetrainResponse()

        try:
            model_info = trained_models[request.model_id]
            data = json.loads(request.data_json)
            hyperparams = {k: float(v) if '.' in v else int(v) for k, v in request.hyperparams.items()}

            # заново обучаем модель, но сохраняем под тем же ID
            model, y_pred, y_test = trainer(
                data, None, model_info["model_name"], hyperparams
            )

            metrics_result = calculate_metrics(model_info["model_name"], y_test, y_pred)

            # обновляем модель в хранилище
            trained_models[request.model_id] = {
                "model": model,
                "model_name": model_info["model_name"],
                "hyperparams": hyperparams,
                "training_time": datetime.now(),
                "metrics": metrics_result
            }

            metrics_proto = {k: float(v) for k, v in metrics_result.items()}

            return ml_service_pb2.RetrainResponse(
                model_id=request.model_id,
                model_name=model_info["model_name"],
                metrics=metrics_proto,
                message="Model retrained successfully"
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.RetrainResponse()

    def DeleteModel(self, request, context):
        # удаляем модель из хранилища
        if request.model_id not in trained_models:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return ml_service_pb2.DeleteModelResponse()

        del trained_models[request.model_id]
        return ml_service_pb2.DeleteModelResponse(
            message=f"Model {request.model_id} deleted successfully"
        )


def serve():
    # настраиваем и запускаем gRPC сервер
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
