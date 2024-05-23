# Work in progress
import threading, gc
from typing import Any, Dict, Optional
from dataclasses import dataclass
import json
import humanfirst

# from .models.huggingface.intent_entity import IntentEntityPipeline, Trainer

from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2, models_pb2_grpc
from .humanfirst.protobuf.external_nlu.v1alpha1 import service_pb2, service_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2
from hf_integration.clu_apis import clu_apis

MODEL_HANDLE_PATH = "/home/fayaz/hf-custom-integration/hf_integration/data/handlemap.json"

@dataclass
class IntegrationServiceConfig:
    max_concurrent_train: int = 1
    max_concurrent_models: int = 2

class ModelService(discovery_pb2_grpc.DiscoveryServicer, models_pb2_grpc.ModelsServicer):
    def __init__(self, config: Optional[IntegrationServiceConfig] = None) -> None:
        super().__init__()
        self.clu_api = clu_apis(clu_endpoint="https://hf-clu-east-us.cognitiveservices.azure.com/",
                                clu_key="")
        self.config = config if config is not None else IntegrationServiceConfig()
        self.next_handle = 0

        with open(MODEL_HANDLE_PATH, mode="r", encoding="utf8") as f:
            self.handle_map = json.load(f)
            print(self.handle_map)

        self.sema_models = threading.BoundedSemaphore(self.config.max_concurrent_train)
        self.sema_train = threading.BoundedSemaphore(self.config.max_concurrent_models)

    def _retain_model(self, model: None) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.handle_map[handle] = model

        return handle

    def _get_model(self, handle: int) -> Any:
        model = self.handle_map.get(handle)
        if model is None:
            raise RuntimeError("no such handle exists")
        return model

    def _flip_dict(self, input_dict, delimiter):
        # Ensure that all values in the original dictionary are unique
        if len(input_dict) != len(set(input_dict.values())):
            raise ValueError("Intents in HF have duplicate names")

        # Flip the dictionary
        flipped_dict = {}
        for key, value in input_dict.items():
            flipped_dict[value] = [key, value.split(delimiter)[-1]]
        return flipped_dict


    def ListModels(self, request: models_pb2.ListModelsRequest, context) -> models_pb2.ListModelsResponse:
        # Indicate that this service does not have any prebuilt models
        print("ListModels")
        return models_pb2.ListModelsResponse()

    def GetModel(self, request: models_pb2.GetModelRequest, context) -> models_pb2.Model:
        # TODO: Implement model store
        print("GetModel")
        raise NotImplementedError()

    def GetTrainParameters(self, request: models_pb2.GetTrainParametersRequest, context) -> models_pb2.GetTrainParametersResponse:
        # Indicate the data format in which trainin data should be provided
        print("GetTrainParameters")
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
        print("TrainModel")
        project_name = "custnluintg345_20240523114146"
        model_label='new_model12345'
        train_split = 100
        deployment_name = "first_deployment"
        self.clu_api.model_train(project_name=project_name,
                                 model_label=model_label,
                                 train_split=train_split)
        
        response = self.clu_api.deploy_trained_model(
                    project_name = project_name,
                    deployment_name = deployment_name,
                    model_label = model_label
                )
        response = response.result()

        handle = response["modelId"]
        self.handle_map[handle] = {
            "project_name": project_name,
            "deployment_name": deployment_name,
            "model_label": model_label
        }

        with open(MODEL_HANDLE_PATH, mode="w", encoding="utf8") as f:
            json.dump(self.handle_map,f,indent=2)

        print(self.handle_map)


        return models_pb2.TrainModelResponse(
            model=models_pb2.Model(
                id=f'{handle}',
                display_name=deployment_name,
                classification=models_pb2.ClassificationConfig(
                    # Number of examples that will be sent in each `Classify` request
                    max_batch_size=1,
                )
            )
        )


    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        print("UnloadModel")
        project_name = "custnluintg345_20240523114146"
        model_label='new_model12345'
        self.clu_api.delete_trained_model(
            project_name = project_name,
            model_label = model_label)
        return models_pb2.UnloadModelResponse()

    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        print("DeleteModel")
        self._release_model(int(request.model_id))
        del self.handle_map[int(request.model_id)]
        gc.collect()
        return models_pb2.DeleteModelResponse()

    def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        print("Classify")
        model_id = request.model_id
        clu_endpoint="https://hf-clu-east-us.cognitiveservices.azure.com/"
        username = "fayaz+my_org@humanfirst.ai"
        password = ""
        namespace = "fayaz"
        playbook = "playbook-GEKT4ZYIV5D7LPSBK7UBHKRY"
        delimiter = "-"
        hf_api = humanfirst.apis.HFAPI(username, password)
        # get all intents
        all_intents = hf_api.get_intents(namespace, playbook)
        # print(json.dumps(all_intents,indent=2))
        hf_json = {
        "$schema": "https://docs.humanfirst.ai/hf-json-schema.json",
        "intents": all_intents
        }

        hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json,delimiter)
        print(hf_workspace.get_intent_index(delimiter=delimiter))
        intent_index = self._flip_dict(hf_workspace.get_intent_index(delimiter=delimiter),delimiter=delimiter)

        predictions = []
        for ex in request.examples:
            print(ex)
            data = self.clu_api.predict(
                        project_name=self.handle_map[model_id]["project_name"],
                        deployment_name=self.handle_map[model_id]["deployment_name"],
                        endpoint=clu_endpoint,
                        text = ex.contents).json()

            # Extract intents and entities
            intents = data['result']['prediction']['intents']
            entities = data['result']['prediction']['entities']

            # Prepare the Go structures
            go_struct = {
                "Predictions": {
                    "matches": [],
                    "entity_matches": []
                }
            }

            # Process intents
            for intent in intents:
                go_struct["Predictions"]["matches"].append(
                    service_pb2.IntentMatch(
                        id=intent_index[intent["category"]][0],
                        name=intent_index[intent["category"]][0],
                        score=intent["confidenceScore"]
                    )
                )

            # Process entities
            for entity in entities:
                entity_ref = service_pb2.EntityReference(
                    key=entity["category"],
                    text=entity["text"],
                    value=entity.get("extraInformation", [{}])[0].get("key", "")
                )
                span = service_pb2.Span(
                    start=entity["offset"],
                    end=entity["offset"] + entity["length"]
                )
                go_struct["Predictions"]["entity_matches"].append(
                    service_pb2.EntityMatch(
                        entity=entity_ref,
                        score=entity["confidenceScore"],
                        span=span,
                    # "extractor": "N/A"  # Assuming extractor information is not provided
                    )
                )

            predictions.append(service_pb2.Predictions(
                matches=go_struct["Predictions"]["matches"],
                entity_matches=go_struct["Predictions"]["entity_matches"]))
            # print(predictions)

        return models_pb2.ClassifyResponse(predictions=predictions)

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        print("Embed")
        raise NotImplementedError()
