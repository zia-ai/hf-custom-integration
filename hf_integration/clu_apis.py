"""
python custom_integration.py
"""
# *********************************************************************************************************************

# standard imports
import re
import json

# 3rd party imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.core.rest import HttpRequest


API_VERSION="2023-04-01"

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
                           des: str) -> None:
        """Create a new project
        Supports English language en-us
        Project kind supported is conversation
        Disabled multilingual
        Project settings
            Threshold: 0.5
            NormalizeCasing: False // This preserves the case of the utterances
        """
        project_name = self._remove_non_alphanumeric(input_string=project_name)
        project = {
            "projectName": project_name,
            "language": "en-us",
            "settings": {
                "confidenceThreshold": 0.5,
                "normalizeCasing": False
            },
            "projectKind": "Conversation",
            "description": des,
            "multilingual": False
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
                    train_split: int = 80):
        """Model Train"""

        test_split = 100 - train_split
        response =  self.client.begin_train(
                    project_name=project_name,
                    configuration = {
                            "modelLabel": model_label,
                            "trainingMode": "standard",
                            "evaluationOptions": {
                                "kind": "percentage",
                                "testingSplitPercentage": test_split,
                                "trainingSplitPercentage": train_split
                            }
                        },
                        content_type = "application/json")
        training_job_status = response.status()
        print(training_job_status)
        print(response.result())
        return response

    def list_trained_model(self,
                           project_name: str):
        """List trained model"""

        response = self.client.list_trained_models(project_name=project_name)

        return response
    
    def delete_trained_model(self,
                             project_name: str,
                             model_label: str):
        """Delete trained model"""

        self.client.delete_trained_model(
            project_name=project_name,
            trained_model_label=model_label
        )
    
    def deploy_trained_model(self,
                project_name: str,
                deployment_name: str,
                model_label: str):
        """Deploy trained model"""

        response = self.client.begin_deploy_project(
            project_name=project_name,
            deployment_name=deployment_name,
            deployment={
                        "trainedModelLabel": model_label
                    },
            content_type = "application/json"
        )
        print(response.result())
        return response

    def get_trained_model(self,
                          project_name: str,
                          model_label: str):
        """Get trained model"""

        response = self.client.get_trained_model(
                    project_name=project_name,
                    trained_model_label=model_label
        )

        print(response)
        return response

    def delete_trained_model(self,
                             project_name: str,
                             model_label: str):
        """Delete trained model"""

        self.client.delete_trained_model(
            project_name=project_name,
            trained_model_label=model_label
        )

    def predict(self,
                project_name: str,
                deployment_name: str,
                endpoint: str,
                text: str):
        """Predict"""

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

        response = self.client.send_request(
            HttpRequest(
                method="POST",
                url=f'{endpoint}/language/:analyze-conversations?api-version={API_VERSION}',
                data=json.dumps(data),  # Convert dictionary to JSON string
                headers={"Content-Type": "application/json"}  # Specify the content type
            )
        )

        response.raise_for_status()  # raises an error if response had error status code

        return response