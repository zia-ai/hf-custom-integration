"""
python custom_integration.py
"""
# *********************************************************************************************************************

# standard imports
import re
import json
import asyncio
from time import sleep
import threading
import logging
import logging.config
import os
from datetime import datetime

# 3rd party imports
import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.core.rest import HttpRequest
from azure.core.exceptions import HttpResponseError
from azure.core import exceptions

API_VERSION="2023-04-01"

# locate where we are
here = os.path.abspath(os.path.dirname(__file__))

path_to_log_config_file = os.path.join(here,'config','logging.conf')

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the log file name with the current datetime
log_filename = f"log_{current_datetime}.log"

# Decide whether to save logs in a file or not
log_file_enable = os.environ.get("CI_LOG_FILE_ENABLE")

log_handler_list = []

if log_file_enable == "TRUE":
    log_handler_list.append('rotatingFileHandler')
elif log_file_enable == "FALSE" or log_file_enable is None:
    pass
else:
    raise RuntimeError("Incorrect CI_LOG_FILE_ENABLE value. Should be - 'TRUE', 'FALSE' or ''")

log_defaults = {}

# get log directory if going to save the logs
if log_file_enable == "TRUE":
    log_dir = os.path.join(here,"logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_to_save_log = os.path.join(log_dir,log_filename)
else:
    # avoid logging to a file
    path_to_save_log = '/dev/null'  # On Linux/MacOS, this discards logs (Windows: NUL) pylint:disable=invalid-name
log_defaults['CI_LOG_FILE_PATH'] = path_to_save_log


# Decide whether to print the logs in the console or not
log_console_enable = os.environ.get("CI_LOG_CONSOLE_ENABLE")

if log_console_enable == "TRUE":
    log_handler_list.append('consoleHandler')
elif log_console_enable == "FALSE" or log_console_enable is None:
    pass
else:
    raise RuntimeError("Incorrect CI_LOG_CONSOLE_ENABLE value. Should be - 'TRUE', 'FALSE' or ''")


if log_console_enable == "TRUE" and log_file_enable == "TRUE":
    raise RuntimeError("Custom integration supports either console logging or file logging but not both")
    # this is because of unable to override SSL errors logging configurations and able to only have them in either console or log file 


if log_handler_list:
    log_defaults['CI_LOG_HANDLER'] = ",".join(log_handler_list)
else:
    log_defaults['CI_LOG_HANDLER'] = "nullHandler"


# Set log levels
log_level = os.environ.get("CI_LOG_LEVEL")
if log_level is not None:
    # set log level
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise RuntimeError("Incorrect log level. Should be - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

    log_defaults['CI_LOG_LEVEL'] = log_level
else:
    log_defaults['CI_LOG_LEVEL'] = 'INFO' # default level


# Load logging configuration
logging.config.fileConfig(
    path_to_log_config_file,
    defaults=log_defaults
)

# create logger
logger = logging.getLogger('custom_integration.clu_apis')

class clu_apis:
    """This class demonstrates CLU APIs"""

    def __init__(self, clu_endpoint: str, clu_key: str) -> None:
        """Authorization"""

        self.client = ConversationAuthoringClient(
            clu_endpoint, AzureKeyCredential(clu_key)
        )

    def get_project_metadata(self,
                             project_name: str):
        """Gets project metadata"""

        return self.client.get_project(project_name=project_name)

    def list_projects(self) -> list:
        """Returns list of projects"""

        list_projects_response = self.client.list_projects()

        list_projects = []
        for project in list_projects_response:
            list_projects.append(project["projectName"])

        return list_projects
    
    def _remove_non_alphanumeric(self,
                                 input_string: str) -> str:
        """Define regex pattern to match non-alphanumeric characters"""
        input_string = input_string.replace(" ","_")
        pattern = re.compile(r'[^a-zA-Z0-9_]')
        # Use sub() function to replace non-alphanumeric characters with an empty string
        result = re.sub(pattern, '', input_string)
        return result

    def clu_create_project(self,
                           project_name: str,
                           des: str,
                           language: str = "en-us",
                           confidence_threshold: float = 0.5,
                           normalize_casing: bool = False,
                           project_kind: str = "Conversation",
                           multilingual: bool = False
                           ) -> None:
        """Create a new project
        Supports English language en-us
        Project kind supported is conversation
        Disabled multilingual
        Project settings
            Threshold: 0.5
            NormalizeCasing: False // This preserves the case of the utterances
        """

        print("CREATE PROJECT API CALL")
        project_name = self._remove_non_alphanumeric(input_string=project_name)
        project = {
            "projectName": project_name,
            "language": language,
            "settings": {
                "confidenceThreshold": confidence_threshold,
                "normalizeCasing": normalize_casing
            },
            "projectKind": project_kind,
            "description": des,
            "multilingual": multilingual
        }
        self.client.create_project(project_name=project_name,project=project)

    def import_project(self,
                       clu_json: dict,
                       project_name: str) -> None:
        """Imports CLU project into CLU"""

        poller = self.client.begin_import_project(
            project_name=project_name,
            project=clu_json
        )
        response = poller.result()
        if response["status"] == "succeeded":
            print(f"Import jobid: {response['jobId']} is successful")
        else:
            raise RuntimeError(f"Import jobid: {response['jobId']} failed")

    def export_project(self,
                       project_name: str):
        """Exports CLU project from microsoft CLU"""

        export_project = self.client.begin_export_project(project_name=project_name,
                                                    string_index_type="Utf16CodeUnit")

        response = self.client.send_request(HttpRequest(method="GET",url=export_project.result()["resultUrl"]))
        response.raise_for_status() # raises an error if response had error status code

        print(f"Export is done successfully")

        return response.json()
    
    def model_train(self,
                    project_name: str,
                    model_label: str = "trained_model",
                    train_split: int = 80,
                    training_mode: str = "standard"):
        """Model Train"""

        print("\nTRAIN MODEL API CALL\n")

        while True:
            try:
                test_split = 100 - train_split
                response =  self.client.begin_train(
                            project_name=project_name,
                            configuration = {
                                    "modelLabel": model_label,
                                    "trainingMode": training_mode,
                                    "evaluationOptions": {
                                        "kind": "percentage",
                                        "testingSplitPercentage": test_split,
                                        "trainingSplitPercentage": train_split
                                    }
                                },
                                content_type = "application/json")
                training_status = response.status()
                print(f"Train Model Status: {training_status}")
                print(f"Response: {response}")
                while training_status in ["InProgress","notStarted","running"]:
                    sleep(5)
                    training_status = response.status()
                    print(f"Train Model Status: {training_status}")
                else:
                    if training_status == "succeeded":
                        response_result = response.result()
                        print(f"Result: {response_result}")
                        return response_result
                    elif training_status == "cancelling":
                        while training_status != "cancelled":
                            sleep(5)
                            training_status = response.status()
                            print(f"Train Model Status: {training_status}")
                        raise exceptions.ServiceResponseError(f'Model training is terminated in CLU.')
                    elif training_status == "Failed":
                        raise RuntimeError(f'Getting Failed status. Agent created might have been deleted')
                    else:
                        raise RuntimeError("Unknown train status from train model")

            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                else:
                    raise

    def list_trained_model(self,
                           project_name: str):
        """List trained model"""

        print("\nLIST TRAINED MODEL API CALL\n")

        while True:
            try:
                response = self.client.list_trained_models(project_name=project_name)
                return response
            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                else:
                    raise

    def delete_trained_model(self,
                             project_name: str,
                             model_label: str):
        """Delete trained model"""

        print("\nDELETE TRAINED MODEL API CALL\n")

        while True:
            try:
                self.client.delete_trained_model(
                    project_name=project_name,
                    trained_model_label=model_label
                )
                return
            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                else:
                    raise
    
    def deploy_trained_model(self,
                project_name: str,
                deployment_name: str,
                model_label: str):
        """Deploy trained model"""

        print("\nDEPLOY MODEL API CALL\n")

        while True:
            try:
                response = self.client.begin_deploy_project(
                    project_name=project_name,
                    deployment_name=deployment_name,
                    deployment={
                                "trainedModelLabel": model_label
                            },
                    content_type = "application/json"
                )
                deploy_status = response.status()
                print(f"Deploy Model Status: {deploy_status}")
                print(f"Response: {response}")
                while deploy_status in ["InProgress","notStarted","running"]:
                    sleep(5)
                    deploy_status = response.status()
                    print(f"Deploy Model Status: {deploy_status}")
                else:
                    if deploy_status == "succeeded":
                        response_result = response.result()
                        print(f"Result: {response_result}")
                        return response_result
                    elif deploy_status == "cancelling":
                        while deploy_status != "cancelled":
                            sleep(5)
                            deploy_status = response.status()
                            print(f"Deploy Model Status: {deploy_status}")
                        raise RuntimeError(f'Model deployment is terminated in CLU.')
                    elif deploy_status == "Failed":
                        raise RuntimeError(f'Getting Failed status. Agent created might have been deleted')
                    else:
                        raise RuntimeError("Unknown deployment status from deploy model")

            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                else:
                    raise

    def get_trained_model(self,
                          project_name: str,
                          model_label: str):
        """Get trained model"""

        print("\nGET TRAINED MODEL API CALL\n")

        while True:
            try:
                response = self.client.get_trained_model(
                            project_name=project_name,
                            trained_model_label=model_label
                )
                return response
            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                else:
                    raise

    async def predict(self, project_name: str, deployment_name: str, endpoint: str, text: str, is_cancelled: threading.Event):
        """Predict"""
        print("\nPREDICT API CALL\n")

        data = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "ID1",
                    "participantId": "ID1",
                    "text": text
                }
            },
            "parameters": {
                "projectName": project_name,
                "deploymentName": deployment_name,
                "stringIndexType": "TextElement_V8"
            }
        }

        request = HttpRequest(
            method="POST",
            url=f'{endpoint}/language/:analyze-conversations?api-version={API_VERSION}',
            data=json.dumps(data),
            headers={"Content-Type": "application/json"}
        )

        while True:
            try:
                # Get the current event loop and send the request using run_in_executor for non-blocking I/O
                loop = asyncio.get_event_loop()

                # Check for cancellation before starting the request
                if asyncio.current_task().cancelled() or is_cancelled.is_set():
                    print("Cancellation requested before predict call.")
                    raise asyncio.CancelledError

                # run_in_executor also raises asyncio.CancelledError error if the task is cancelled
                response = await loop.run_in_executor(None, self.client.send_request, request)
                response.raise_for_status()
                return response.json()
            
            except HttpResponseError as e:
                if e.response.status_code == 429:
                    print("Caught 429 Response code")
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")

                    # Check for cancellation before sleeping for retry
                    if asyncio.current_task().cancelled() or is_cancelled.is_set():
                        print("Cancellation requested during retry wait.")
                        raise asyncio.CancelledError

                    await asyncio.sleep(retry_after)
                else:
                    raise # Raises other exceptions such as resource not found (deployment doesn't exists), etc., which handling is not required
                    # do not return any empty values while trying to handle any exceptions. Might cause issues while handling the aynchronously generated results. 
            except asyncio.CancelledError:
                print("Predict task has been cancelled.")
                raise

    def delete_project(self,
                       project_name: str):
        """Deletes a project"""

        print("\nDELETE PROJECT API CALL\n")

        while True:
            try:
                response = self.client.begin_delete_project(project_name=project_name)
                return response
            except HttpResponseError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    sleep(retry_after)
                elif e.response.status_code == 404:
                    return response
                else:
                    print("Not handling this reponse code")
                    raise