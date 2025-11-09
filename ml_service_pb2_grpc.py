"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import ml_service_pb2 as ml__service__pb2


class MLServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        # настраиваем все методы, которые можно вызывать на сервере
        # каждый метод знает, как сериализовать запрос и десериализовать ответ
        self.HealthCheck = channel.unary_unary(
                '/mlservice.MLService/HealthCheck',
                request_serializer=ml__service__pb2.HealthRequest.SerializeToString,
                response_deserializer=ml__service__pb2.HealthResponse.FromString,
                )
        self.ListModels = channel.unary_unary(
                '/mlservice.MLService/ListModels',
                request_serializer=ml__service__pb2.ListModelsRequest.SerializeToString,
                response_deserializer=ml__service__pb2.ListModelsResponse.FromString,
                )
        self.GetModelParams = channel.unary_unary(
                '/mlservice.MLService/GetModelParams',
                request_serializer=ml__service__pb2.GetModelParamsRequest.SerializeToString,
                response_deserializer=ml__service__pb2.GetModelParamsResponse.FromString,
                )
        self.TrainModel = channel.unary_unary(
                '/mlservice.MLService/TrainModel',
                request_serializer=ml__service__pb2.TrainRequest.SerializeToString,
                response_deserializer=ml__service__pb2.TrainResponse.FromString,
                )
        self.ListTrainedModels = channel.unary_unary(
                '/mlservice.MLService/ListTrainedModels',
                request_serializer=ml__service__pb2.ListTrainedModelsRequest.SerializeToString,
                response_deserializer=ml__service__pb2.ListTrainedModelsResponse.FromString,
                )
        self.Predict = channel.unary_unary(
                '/mlservice.MLService/Predict',
                request_serializer=ml__service__pb2.PredictRequest.SerializeToString,
                response_deserializer=ml__service__pb2.PredictResponse.FromString,
                )
        self.RetrainModel = channel.unary_unary(
                '/mlservice.MLService/RetrainModel',
                request_serializer=ml__service__pb2.RetrainRequest.SerializeToString,
                response_deserializer=ml__service__pb2.RetrainResponse.FromString,
                )
        self.DeleteModel = channel.unary_unary(
                '/mlservice.MLService/DeleteModel',
                request_serializer=ml__service__pb2.DeleteModelRequest.SerializeToString,
                response_deserializer=ml__service__pb2.DeleteModelResponse.FromString,
                )


class MLServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    # это заготовки методов, которые нужно реализовать в настоящем сервере
    # по умолчанию они просто возвращают ошибку "не реализовано"

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelParams(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TrainModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListTrainedModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RetrainModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MLServiceServicer_to_server(servicer, server):
    # здесь мы связываем наши методы с gRPC сервером
    # для каждого метода указываем, как обрабатывать запросы и ответы
    rpc_method_handlers = {
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=ml__service__pb2.HealthRequest.FromString,
                    response_serializer=ml__service__pb2.HealthResponse.SerializeToString,
            ),
            'ListModels': grpc.unary_unary_rpc_method_handler(
                    servicer.ListModels,
                    request_deserializer=ml__service__pb2.ListModelsRequest.FromString,
                    response_serializer=ml__service__pb2.ListModelsResponse.SerializeToString,
            ),
            'GetModelParams': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelParams,
                    request_deserializer=ml__service__pb2.GetModelParamsRequest.FromString,
                    response_serializer=ml__service__pb2.GetModelParamsResponse.SerializeToString,
            ),
            'TrainModel': grpc.unary_unary_rpc_method_handler(
                    servicer.TrainModel,
                    request_deserializer=ml__service__pb2.TrainRequest.FromString,
                    response_serializer=ml__service__pb2.TrainResponse.SerializeToString,
            ),
            'ListTrainedModels': grpc.unary_unary_rpc_method_handler(
                    servicer.ListTrainedModels,
                    request_deserializer=ml__service__pb2.ListTrainedModelsRequest.FromString,
                    response_serializer=ml__service__pb2.ListTrainedModelsResponse.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=ml__service__pb2.PredictRequest.FromString,
                    response_serializer=ml__service__pb2.PredictResponse.SerializeToString,
            ),
            'RetrainModel': grpc.unary_unary_rpc_method_handler(
                    servicer.RetrainModel,
                    request_deserializer=ml__service__pb2.RetrainRequest.FromString,
                    response_serializer=ml__service__pb2.RetrainResponse.SerializeToString,
            ),
            'DeleteModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteModel,
                    request_deserializer=ml__service__pb2.DeleteModelRequest.FromString,
                    response_serializer=ml__service__pb2.DeleteModelResponse.SerializeToString,
            ),
    }
    # регистрируем все обработчики на сервере под именем нашего сервиса
    generic_handler = grpc.method_handlers_generic_handler(
            'mlservice.MLService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 class MLService(object):
    """Missing associated documentation comment in .proto file."""

    # это статические методы для удобного вызова сервера из клиентского кода
    # они скрывают всю сложность работы с gRPC каналами

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/HealthCheck',
            ml__service__pb2.HealthRequest.SerializeToString,
            ml__service__pb2.HealthResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/ListModels',
            ml__service__pb2.ListModelsRequest.SerializeToString,
            ml__service__pb2.ListModelsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModelParams(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/GetModelParams',
            ml__service__pb2.GetModelParamsRequest.SerializeToString,
            ml__service__pb2.GetModelParamsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TrainModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/TrainModel',
            ml__service__pb2.TrainRequest.SerializeToString,
            ml__service__pb2.TrainResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListTrainedModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/ListTrainedModels',
            ml__service__pb2.ListTrainedModelsRequest.SerializeToString,
            ml__service__pb2.ListTrainedModelsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/Predict',
            ml__service__pb2.PredictRequest.SerializeToString,
            ml__service__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RetrainModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/RetrainModel',
            ml__service__pb2.RetrainRequest.SerializeToString,
            ml__service__pb2.RetrainResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mlservice.MLService/DeleteModel',
            ml__service__pb2.DeleteModelRequest.SerializeToString,
            ml__service__pb2.DeleteModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
