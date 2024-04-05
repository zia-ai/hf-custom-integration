"""
python custom_integration.py
"""
# *********************************************************************************************************************

# standard imports
import re

# 3rd party imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.core.rest import HttpRequest

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
