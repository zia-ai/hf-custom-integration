"""
Handles workspaces during import and export
"""

# standard import
import json
import gzip
import io

# custom imports
from hf_integration.workspace_generic import WorkspaceServiceGeneric
from hf_integration.clu_apis import clu_apis
from hf_integration.clu_converters import clu_converter
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2, workspace_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

class WorkspaceServiceCLU(WorkspaceServiceGeneric):
    """
    This is a workspace service for Microsoft CLU
    """

    def __init__(self, config: dict) -> None:
        """Authorization"""

        super().__init__(config)
        self.clu_api = clu_apis(clu_endpoint=self.config["clu_endpoint"],
                                clu_key=self.config["clu_key"])
        self.clu_converter = clu_converter()

    def ListWorkspaces(self, request: workspace_pb2.ListWorkspacesRequest, context) -> workspace_pb2.ListWorkspacesResponse:
        """List Workspaces"""

        workspaces = []
        print("CLU ListWorkspace")
        for project in self.clu_api.list_projects():
                workspaces.append(workspace_pb2.Workspace(id=project, name=project))

        return workspace_pb2.ListWorkspacesResponse(workspaces=workspaces)

    def GetWorkspace(self, request: workspace_pb2.GetWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """Get workspace"""

        if request.workspace_id in self.clu_api.list_projects():
            return workspace_pb2.Workspace(id=request.workspace_id, name=request.workspace_id)
        else:
            raise RuntimeError("no such workspace")

    def CreateWorkspace(self, request: workspace_pb2.CreateWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """
        Create a new workspace
        """

        self.clu_api.clu_create_project(project_name=request.workspace.name,
                                        des = request.workspace.description)

        return workspace_pb2.Workspace(id=request.workspace.name, name=request.workspace.name)


    def GetImportParameters(self, request: workspace_pb2.GetImportParametersRequest, context) -> workspace_pb2.GetImportParametersResponse:
        """
        Indicate the data format in which the workspace data should be provided

        In this case, we specifically request the HF json format
        """

        return workspace_pb2.GetImportParametersResponse(data_format=self.data_format, format_options=self.format_options)

    def ImportWorkspace(self, request: workspace_pb2.ImportWorkspaceRequest, context) -> workspace_pb2.ImportWorkspaceResponse:
        """
        Import a workspace into the integration, from the provided data exported from Studio
        """

        # Write the HF json data to a file (in the real world, you would want to make sure there are no path injection attempts)
        # Decompress the gzip data
        with gzip.open(io.BytesIO(request.data), 'rb') as f:
            uncompressed_data = f.read()
        hf_json = json.loads(uncompressed_data.decode('utf-8'))
        project_name = self.clu_api._remove_non_alphanumeric(input_string=request.workspace_id)
        project_metadata = self.clu_api.get_project_metadata(project_name=project_name)
        clu_json = {
            "projectFileVersion": "2023-04-01",
            "stringIndexType": "Utf16CodeUnit",
            "metadata": {
                "projectKind": project_metadata["projectKind"],
                "settings": project_metadata["settings"],
                "projectName": project_metadata["projectName"],
                "multilingual": project_metadata["multilingual"],
                "description": project_metadata["description"],
                "language": project_metadata["language"]
            },
            "assets": {
                "projectKind": project_metadata["projectKind"],
                "intents": [],
                "entities": [],
                "utterances": []
            }
        }

        clu_json = self.clu_converter.hf_to_clu_process(hf_json, clu_json)

        self.clu_api.import_project(clu_json=clu_json,
                                    project_name=project_name)

        return workspace_pb2.ImportWorkspaceResponse()

    def ExportWorkspace(self, request: workspace_pb2.ExportWorkspaceRequest, context) -> workspace_pb2.ExportWorkspaceResponse:
        """
        Exports a workspace from the integration, importing it into Studio
        """
        
        clu_project = self.clu_api.export_project(project_name=request.workspace_id)

        hf_json = self.clu_converter.clu_to_hf_process(clu_json=clu_project)

        # Read the HF json data from the file
        return workspace_pb2.ExportWorkspaceResponse(
            data_format=config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON,
            format_options=self.format_options,
            data=gzip.compress(json.dumps(hf_json).encode('utf-8')),
        )
