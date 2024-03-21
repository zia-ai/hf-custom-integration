

import threading, gc
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .models.huggingface.intent_entity import IntentEntityPipeline, Trainer

from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2, models_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2


@dataclass
class IntegrationServiceConfig:
    max_concurrent_train: int = 1
    max_concurrent_models: int = 2

class ModelService(discovery_pb2_grpc.DiscoveryServicer, models_pb2_grpc.ModelsServicer):
    def __init__(self, config: Optional[IntegrationServiceConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else IntegrationServiceConfig()
        self.next_handle = 0
        self.handle_map: Dict[int, IntentEntityPipeline] = {}
        self.sema_models = threading.BoundedSemaphore(self.config.max_concurrent_train)
        self.sema_train = threading.BoundedSemaphore(self.config.max_concurrent_models)

    def _retain_model(self, model: IntentEntityPipeline) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.handle_map[handle] = model

        return handle

    def _get_model(self, handle: int) -> Any:
        model = self.handle_map.get(handle)
        if model is None:
            raise RuntimeError("no such handle exists")
        return model

    def _release_model(self, handle):
        model = self.handle_map.get(handle)
        if model is not None:
            self.sema_models.release()

    def ListModels(self, request: models_pb2.ListModelsRequest, context) -> models_pb2.ListModelsResponse:
        # Indicate that this service does not have any prebuilt models
        return models_pb2.ListModelsResponse()

    def GetModel(self, request: models_pb2.GetModelRequest, context) -> models_pb2.Model:
        # TODO: Implement model store
        raise NotImplementedError()

    def GetTrainParameters(self, request: models_pb2.GetTrainParametersRequest, context) -> models_pb2.GetTrainParametersResponse:
        # Indicate the data format in which trainin data should be provided
        return models_pb2.GetTrainParametersResponse(
            data_format=config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON,
            # These are all the default options, except for skip_empty_intents
            format_options=config_pb2.IntentsDataOptions(
                hierarchical_intent_name_disabled=False,
                hierarchical_delimiter="",
                zip_encoding=False,
                gzip_encoding=False,
                hierarchical_follow_up=False,
                include_negative_phrases=False,
                intent_tag_predicate=None,
                phrase_tag_predicate=None,
                skip_empty_intents=True
            )
        )

    def TrainModel(self, request: models_pb2.TrainModelRequest, context) -> models_pb2.TrainModelResponse:
        # Acquire a semaphone in order to restrict the amount of live models in memory
        self.sema_models.acquire()

        try:
            self.sema_train.acquire()
            trainer = Trainer(workspace_data=request.data)
            pipeline = trainer.train()
            handle = self._retain_model(pipeline)

            return models_pb2.TrainModelResponse(
                model=models_pb2.Model(
                    id=f'{handle}',
                    classification=models_pb2.ClassificationConfig(
                        # Number of examples that will be sent in each `Classify` request
                        max_batch_size=1000,
                    )
                )
            )
        except Exception:
            self.sema_models.release()
            raise
        finally:
            self.sema_train.release()

    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        self._release_model(int(request.model_id))
        del self.handle_map[int(request.model_id)]
        gc.collect()
        return models_pb2.UnloadModelResponse()

    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        self._release_model(int(request.model_id))
        del self.handle_map[int(request.model_id)]
        gc.collect()
        return models_pb2.DeleteModelResponse()

    def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        pipeline = self._get_model(int(request.model_id))

        inputs = [ex.contents for ex in request.examples]
        predictions = pipeline(inputs)

        return models_pb2.ClassifyResponse(predictions=predictions)

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        raise NotImplementedError()
