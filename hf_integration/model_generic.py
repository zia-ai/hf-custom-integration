"""
Handles model training and k-fold evaluation
"""

# standard imports
import json
import os

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2_grpc, models_pb2, models_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2


SNAPSHOT_PATH = "/home/fayaz/hf-custom-integration/hf_integration/workspaces/"
MODEL_HANDLE_PATH = "/home/fayaz/hf-custom-integration/hf_integration/data/handlemap.json"

class ModelServiceGeneric(discovery_pb2_grpc.DiscoveryServicer, models_pb2_grpc.ModelsServicer):
    """
    This is a model service that can train and run k-fold evaluation
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_format = config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON
        self.format_options = config_pb2.IntentsDataOptions(
            hierarchical_intent_name_disabled=False,
            hierarchical_delimiter="",
            zip_encoding=False,
            gzip_encoding=True,
            hierarchical_follow_up=False,
            include_negative_phrases=False,
            intent_tag_predicate=None,
            phrase_tag_predicate=None,
            skip_empty_intents=False,
        )

        if not os.path.exists(MODEL_HANDLE_PATH):
            with open(MODEL_HANDLE_PATH, mode="w", encoding="utf8") as f:
                json.dump({},f,indent=2)
        with open(MODEL_HANDLE_PATH, mode="r", encoding="utf8") as f:
            self.handle_map = json.load(f)


    def ListModels(self, request: models_pb2.ListModelsRequest, context) -> models_pb2.ListModelsResponse:
        """Indicate that this service does not have any prebuilt models"""

        return models_pb2.ListModelsResponse()

    def GetModel(self, request: models_pb2.GetModelRequest, context) -> models_pb2.Model:
        """
        If model store is implemented, then this requests gets the model from the model store
        
        Not implemented yet
        """

        raise NotImplementedError()

    def GetTrainParameters(self, request: models_pb2.GetTrainParametersRequest, context) -> models_pb2.GetTrainParametersResponse:
        """Indicate the data format in which training data should be provided"""

        return models_pb2.GetTrainParametersResponse(
            data_format=self.data_format,
            format_options=self.format_options
        )

    def TrainModel(self, request: models_pb2.TrainModelRequest, context) -> models_pb2.TrainModelResponse:
        """Trains a model"""

        return models_pb2.TrainModelResponse(
            model=None
        )


    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        """Unload Model"""

        return models_pb2.UnloadModelResponse()


    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        """Delete Model"""

        return models_pb2.DeleteModelResponse()


    def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        """Predicts utterances"""

        return models_pb2.ClassifyResponse(predictions=None)

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        """
        Embeddings

        Not implemented yet
        """
        
        raise NotImplementedError()
