"""
Handles workspaces during import and export
"""

# standard imports
import json
import datetime
from uuid import uuid4
import os

# 3rd party imports
import humanfirst

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import models_pb2
from .humanfirst.protobuf.external_nlu.v1alpha1 import service_pb2
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2
from hf_integration.clu_apis import clu_apis
from hf_integration.workspace_clu import WorkspaceServiceCLU
from hf_integration.model_generic import ModelServiceGeneric

SNAPSHOT_PATH = "/home/fayaz/hf-custom-integration/hf_integration/workspaces/"
MODEL_HANDLE_PATH = "/home/fayaz/hf-custom-integration/hf_integration/data/handlemap.json"
TRAIN_SPLIT=100
MAX_BATCH_SIZE=1000

class ModelServiceCLU(ModelServiceGeneric):
    """
    This is a model service that can train and run k-fold evaluation
    """

    def __init__(self, config: dict) -> None:
        """Authorization"""

        super().__init__(config)
        self.clu_api = clu_apis(clu_endpoint=self.config["clu_endpoint"],
                                clu_key=self.config["clu_key"])
        self.workspace = WorkspaceServiceCLU(config=config)


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
            # These are all the default options, except for skip_empty_intents
            format_options=self.format_options
        )


    def TrainModel(self, request: models_pb2.TrainModelRequest, context) -> models_pb2.TrainModelResponse:
        """Trains a model"""

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        project_name = self.clu_api._remove_non_alphanumeric(
            input_string=f"agent_{uuid4()}")

        model_label = f"model_{timestamp}"
        
        # TODO: Manual and automatic split implement
        train_split = TRAIN_SPLIT
        deployment_name = f"deployment_{timestamp}"

        # create a new project
        self.clu_api.clu_create_project(project_name=project_name,
                            des = "Train and eval")
        print("\nNew project created")

        # Create import request object
        import_request = workspace_pb2.ImportWorkspaceRequest(
            namespace= request.namespace,
            integration_id=request.integration_id,
            data=request.data,
            workspace_id=project_name
        )

        hf_file_path = os.path.join(SNAPSHOT_PATH,"import",f"{timestamp}_hf_{request.namespace}_{project_name}.json")
        clu_file_path = os.path.join(SNAPSHOT_PATH,"import",f"{timestamp}_clu_{request.namespace}_{project_name}.json")
        
        import_context = {
            "hf_file_path": hf_file_path,
            "clu_file_path": clu_file_path
        }

        self.workspace.ImportWorkspace(request=import_request, context=import_context)
        print("\nProject imported")


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
            "model_label": model_label,
            "timestamp": timestamp,
            "hf_file_path": hf_file_path,
            "clu_file_path": clu_file_path

        }

        with open(MODEL_HANDLE_PATH, mode="w", encoding="utf8") as f:
            json.dump(self.handle_map,f,indent=2)


        return models_pb2.TrainModelResponse(
            model=models_pb2.Model(
                id=f'{handle}',
                display_name=deployment_name,
                classification=models_pb2.ClassificationConfig(
                    # Number of examples that will be sent in each `Classify` request
                    max_batch_size=MAX_BATCH_SIZE,
                )
            )
        )


    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        """Unload Model"""

        project_name = self.handle_map[request.model_id]["project_name"]
        model_label = self.handle_map[request.model_id]["model_label"]
        self.clu_api.delete_trained_model(
            project_name = project_name,
            model_label = model_label)
        return models_pb2.UnloadModelResponse()


    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        """Delete Model"""

        project_name = self.handle_map[request.model_id]["project_name"]
        self.clu_api.delete_project(
            project_name = project_name
        )
        return models_pb2.DeleteModelResponse()


    def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        """Predicts utterances"""

        with open(self.handle_map[request.model_id]["hf_file_path"], mode="r", encoding="utf8") as f:
            hf_json = json.load(f)

        hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json,self.config["delimiter"])
        intent_index = self._flip_dict(hf_workspace.get_intent_index(
            delimiter=self.config["delimiter"]),
            delimiter=self.config["delimiter"]
        )

        predictions = []
        for ex in request.examples:
            print(ex)
            data = self.clu_api.predict(
                        project_name=self.handle_map[request.model_id]["project_name"],
                        deployment_name=self.handle_map[request.model_id]["deployment_name"],
                        endpoint=self.config["clu_endpoint"],
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

        return models_pb2.ClassifyResponse(predictions=predictions)


    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        """
        Embeddings

        Not implemented yet
        """

        raise NotImplementedError()
