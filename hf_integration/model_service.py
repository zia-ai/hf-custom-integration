"""
Handles model training and k-fold evaluation
"""
# standard imports
import asyncio

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import models_pb2
from hf_integration.model_generic import ModelServiceGeneric

class ModelService():
    """
    This is a model service that can train and run k-fold evaluation
    """

    def __init__(self, integration: ModelServiceGeneric) -> None:
        self.integration = integration

    def ListModels(self, request: models_pb2.ListModelsRequest, context) -> models_pb2.ListModelsResponse:
        """Indicate that this service does not have any prebuilt models"""
        
        return self.integration.ListModels(request=request, context=context)

    def GetModel(self, request: models_pb2.GetModelRequest, context) -> models_pb2.Model:
        """
        If model store is implemented, then this requests gets the model from the model store
        
        Not implemented yet
        """

        return self.integration.GetModel(request=request, context=context)

    def GetTrainParameters(self, request: models_pb2.GetTrainParametersRequest, context) -> models_pb2.GetTrainParametersResponse:
        """Indicate the data format in which training data should be provided"""
        
        return self.integration.GetTrainParameters(request=request, context=context)

    def TrainModel(self, request: models_pb2.TrainModelRequest, context) -> models_pb2.TrainModelResponse:
        """Trains a model"""
        
        return self.integration.TrainModel(request=request, context=context)


    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        """Unload Model"""

        return self.integration.UnloadModel(request=request, context=context)


    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        """Delete Model"""

        return self.integration.DeleteModel(request=request, context=context)


    def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        """Predicts utterances"""

        return asyncio.run(self.integration.Classify(request=request, context=context))

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        """
        Embeddings

        Not implemented yet
        """

        return self.integration.Embed(request=request, context=context)
