"""
Handles workspaces during import and export
"""

# standard imports
import json
import datetime
from uuid import uuid4
import os
import asyncio
from time import time, sleep
import threading

# 3rd party imports
import humanfirst
import aiohttp
import grpc

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import models_pb2
from .humanfirst.protobuf.external_nlu.v1alpha1 import service_pb2
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2
from hf_integration.clu_apis import clu_apis
from hf_integration.clu_converters import utf8span
from hf_integration.workspace_clu import WorkspaceServiceCLU
from hf_integration.model_generic import ModelServiceGeneric

TRAIN_SPLIT=100
MAX_BATCH_SIZE=1000
CLU_SUPPORTED_LANGUAGE_CODES = [
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", 
    "cy", "da", "de", "el", "en-us", "en-gb", "eo", "es", "et", "eu", "fa", 
    "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "hu", 
    "hy", "id", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", 
    "la", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", 
    "nl", "nb", "or", "pa", "pl", "ps", "pt-br", "pt-pt", "ro", "ru", "sa", 
    "sd", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", 
    "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "xh", "yi", "zh-hans", 
    "zh-hant", "zu"
]

CLU_SUPPORTED_TRAINING_MODES = ["standard","advanced"]

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

        # Check for language code support
        if self.config["clu_language"] in CLU_SUPPORTED_LANGUAGE_CODES:
            self.language = self.config["clu_language"]
        else:
            raise RuntimeError(f'{self.config["clu_language"]} is not supported by CLU')

        self.multilingual = {"True": True, "False": False}[self.config["clu_multilingual"]]

        # Check for correct training mode
        if self.config["clu_training_mode"] in CLU_SUPPORTED_TRAINING_MODES:
            self.training_mode = self.config["clu_training_mode"]
        else:
            raise RuntimeError(
                f'{self.config["clu_training_mode"]} training mode is not supported. Modes should be {CLU_SUPPORTED_TRAINING_MODES}')
        
        # Check for max_batch_size
        if "max_batch_size" not in self.config:
            self.config["max_batch_size"] = MAX_BATCH_SIZE

        self.config["max_batch_size"] = int(self.config["max_batch_size"])

        if self.config["max_batch_size"] <= 0:
            raise RuntimeError(f'Max Batch Size cannot be less than or qual to 0')


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

        print("ListModels")

        return models_pb2.ListModelsResponse()

    def GetModel(self, request: models_pb2.GetModelRequest, context) -> models_pb2.Model:
        """
        If model store is implemented, then this requests gets the model from the model store
        
        Not implemented yet
        """

        print("GetModel")

        raise NotImplementedError()

    def GetTrainParameters(self, request: models_pb2.GetTrainParametersRequest, context) -> models_pb2.GetTrainParametersResponse:
        """Indicate the data format in which training data should be provided"""

        print("GetTrainParameters")

        return models_pb2.GetTrainParametersResponse(
            data_format=self.data_format,
            # These are all the default options, except for skip_empty_intents
            format_options=self.format_options
        )


    def TrainModel(self, request: models_pb2.TrainModelRequest, context) -> models_pb2.TrainModelResponse:
        """Trains a model and handles cancellation"""

        print("TrainModel")

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        project_name = self.clu_api._remove_non_alphanumeric(
            input_string=f"agent_{uuid4()}")

        model_label = f"model_{timestamp}"

        # TODO: Manual and automatic split implement
        train_split = TRAIN_SPLIT
        deployment_name = f"deployment_{timestamp}"

        # A flag to check for cancellation
        self.is_cancelled_train = threading.Event()

        # A flag to indicate that the request has completed successfully
        self.request_completed_train = threading.Event()

        # Callback to handle cancellation
        def on_cancel():
            if not self.request_completed_train.is_set():
                print("Training has been cancelled.")
                self.is_cancelled_train.set()

                # Cleanup logic here
                if project_name:
                    self.clu_api.delete_project(
                        project_name = project_name)

        # Register cancellation callback
        context.add_callback(on_cancel)
        try:
            # Check for cancellation periodically
            if self.is_cancelled_train.is_set():
                print("Cancelled before project creation.")
                # Send signal to client that the operation has been aborted
                context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
                return

            # Create a new project
            self.clu_api.clu_create_project(project_name=project_name,
                                            des="Train and eval",
                                            language=self.language,
                                            multilingual=self.multilingual)
            print("\nNew project created")

            # Check for cancellation
            if self.is_cancelled_train.is_set():
                print("Cancelled after project creation.")
                # Send signal to client that the operation has been aborted
                context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
                return

            # Create import request object
            import_request = workspace_pb2.ImportWorkspaceRequest(
                namespace=request.namespace,
                integration_id=request.integration_id,
                data=request.data,
                workspace_id=project_name
            )

            hf_file_path = os.path.join(self.snapshot_path, "import", f"{timestamp}_hf_{request.namespace}_{project_name}.json")
            clu_file_path = os.path.join(self.snapshot_path, "import", f"{timestamp}_clu_{request.namespace}_{project_name}.json")

            import_context = {
                "hf_file_path": hf_file_path,
                "clu_file_path": clu_file_path
            }

            # Import workspace
            self.workspace.ImportWorkspace(request=import_request, context=import_context)
            print("\nProject imported")

            # Check for cancellation
            if self.is_cancelled_train.is_set():
                print("Cancelled after workspace import.")
                # Send signal to client that the operation has been aborted
                context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
                return

            # Train the model
            _ = self.clu_api.model_train(project_name=project_name,
                                        model_label=model_label,
                                        train_split=train_split,
                                        training_mode=self.training_mode)

            # Check for cancellation
            if self.is_cancelled_train.is_set():
                print("Cancelled during training.")
                # Send signal to client that the operation has been aborted
                context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
                return

            # Deploy the trained model
            response = self.clu_api.deploy_trained_model(
                project_name=project_name,
                deployment_name=deployment_name,
                model_label=model_label
            )

            handle = response["modelId"]
            self.handle_map[handle] = {
                "project_name": project_name,
                "deployment_name": deployment_name,
                "model_label": model_label,
                "timestamp": timestamp,
                "hf_file_path": hf_file_path,
                "clu_file_path": clu_file_path
            }

            with open(self.model_handle_path, mode="w", encoding="utf8") as f:
                json.dump(self.handle_map, f, indent=2)
            
            # Check for cancellation
            if self.is_cancelled_train.is_set():
                print("Cancelled during deployment.")
                # Send signal to client that the operation has been aborted
                context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
                return

            # Mark the request as completed, so the callback won't trigger cancellation logic
            self.request_completed_train.set()

            # Return the response after successful training and deployment
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

        # Catch RpcError exceptions which may be raised if context is cancelled
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                print("Caught cancellation during processing.")
                return context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
            else:
                # Handle other gRPC exceptions if necessary
                print(f"An error occurred: {e}")
                context.abort(grpc.StatusCode.UNKNOWN, "An unknown error occurred during training.")

    def UnloadModel(self, request: models_pb2.UnloadModelRequest, context) -> models_pb2.UnloadModelResponse:
        """Unload Model"""

        print("UnloadModel")

        project_name = self.handle_map[request.model_id]["project_name"]
        model_label = self.handle_map[request.model_id]["model_label"]
        self.clu_api.delete_trained_model(
            project_name = project_name,
            model_label = model_label)
        return models_pb2.UnloadModelResponse()


    def DeleteModel(self, request: models_pb2.DeleteModelRequest, context) -> models_pb2.DeleteModelResponse:
        """Delete Model"""

        print("DeleteModel")

        project_name = self.handle_map[request.model_id]["project_name"]
        self.clu_api.delete_project(
            project_name = project_name
        )
        return models_pb2.DeleteModelResponse()


    async def Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        """Predicts utterances"""

        print("Classify")

        # Set up cancellation flag
        self.is_cancelled_classify = threading.Event()

        # A flag to indicate that the request has completed successfully
        self.request_completed_classify = threading.Event()

        async def cancel_all_tasks():
            for task in asyncio.all_tasks():
                task.cancel()
                # asyncio signals the task to stop, but it doesn't immediately halt the task's execution at the exact moment it's called.
                # Instead, task.cancel() raises a CancelledError within the task,
                # which allows the task to either handle the cancellation gracefully or let it propagate and terminate.

        # Define the cancellation callback
        def on_cancel():
            if not self.request_completed_classify.is_set():
                print("Inference has been cancelled.")
                self.is_cancelled_classify.set()
                project_name = self.handle_map[request.model_id]["project_name"]

                try:
                    # Get the current event loop of the main thread or create one
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no loop is found (because it's running in a different thread)
                    print("No running event loop, creating a new one.")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Post the cancellation task to the main event loop
                loop.call_soon_threadsafe(cancel_all_tasks)  # Schedules canceling of tasks asynchronously

                # Cleanup logic here
                project_list = self.clu_api.list_projects()
                if project_name in project_list:
                    if project_name:
                        self.clu_api.delete_project(
                            project_name = project_name)

        # Register the cancellation callback
        context.add_callback(on_cancel)

        return await self._Classify(request=request, context=context)

    async def _Classify(self, request: models_pb2.ClassifyRequest, context) -> models_pb2.ClassifyResponse:
        """Predicts utterances"""

        try:
            # Open the model data
            with open(self.handle_map[request.model_id]["hf_file_path"], mode="r", encoding="utf8") as f:
                hf_json = json.load(f)

            hf_workspace = humanfirst.objects.HFWorkspace.from_json(hf_json, self.config["delimiter"])
            intent_index = self._flip_dict(hf_workspace.get_intent_index(
                delimiter=self.config["delimiter"]),
                delimiter=self.config["delimiter"]
            )
            predictions = []
            predict_results = []
            tasks = []

            for ex in request.examples:
                # Check for cancellation before creating tasks
                if self.is_cancelled_classify.is_set():
                    print("Cancelled before inference tasks.")
                    context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled by client.")
                    return

                # Add inference tasks
                tasks.append(
                    self.clu_api.predict(
                        project_name=self.handle_map[request.model_id]["project_name"],
                        deployment_name=self.handle_map[request.model_id]["deployment_name"],
                        endpoint=self.config["clu_endpoint"],
                        text=ex.contents,
                        is_cancelled = self.is_cancelled_classify
                    )
                )

            # Perform the predictions, checking for cancellation
            try:
                predict_results = await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                print("Cancelled during async gather.")
                context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled during processing.")
                return

            # Check if cancellation happened after prediction results
            if self.is_cancelled_classify.is_set():
                print("Cancelled after predictions.")
                context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled by client.")
                return

            for data in predict_results:
                # Do not uncomment
                # if isinstance(data, Exception):
                #     print(f"Skipping failed task due to error: {data}")
                #     continue
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
                    start_char_index=entity["offset"][0] if isinstance(entity["offset"],tuple) else entity["offset"]
                    end_char_index=entity["offset"] + entity["length"]
                    start_byte_index = utf8span(data["result"]["query"],start_char_index)
                    end_byte_index = utf8span(data["result"]["query"],end_char_index)
                    span = service_pb2.Span(
                        start=start_byte_index,
                        end=end_byte_index
                    )
                    go_struct["Predictions"]["entity_matches"].append(
                        service_pb2.EntityMatch(
                            entity=entity_ref,
                            score=entity["confidenceScore"],
                            span=span,
                        )
                    )

                predictions.append(service_pb2.Predictions(
                    matches=go_struct["Predictions"]["matches"],
                    entity_matches=go_struct["Predictions"]["entity_matches"]))

                # Check if cancellation happened while processing prediction results
                if self.is_cancelled_classify.is_set():
                    print("Cancelled while processing prediction results.")
                    context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled by client.")
                    return

            # Check if cancellation happened while processing prediction results
            if self.is_cancelled_classify.is_set():
                print("Cancelled after processing prediction results")
                context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled by client.")
                return
            
            # Mark the request as completed, so the callback won't trigger cancellation logic
            self.request_completed_classify.set()

            return models_pb2.ClassifyResponse(predictions=predictions)

        # Catch RpcError exceptions which may be raised if context is cancelled
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                print("Caught cancellation during processing.")
                return context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.")
            else:
                # Handle other gRPC exceptions if necessary
                print(f"An error occurred: {e}")
                context.abort(grpc.StatusCode.UNKNOWN, "An unknown error occurred during training.")
        except asyncio.CancelledError:
            print("Cancelled during processing.")
            context.abort(grpc.StatusCode.CANCELLED, "Inference cancelled by client.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal error occurred.")

    def Embed(self, request: models_pb2.EmbedRequest, context) -> models_pb2.EmbedResponse:
        """
        Embeddings

        Not implemented yet
        """

        raise NotImplementedError()
