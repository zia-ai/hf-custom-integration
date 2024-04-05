"""
Handles workspaces during import and export
"""

# standard imports
import os
import gzip
import json
import io

# custom imports
from hf_integration.workspace_generic import WorkspaceServiceGeneric
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2, workspace_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

class WorkspaceServiceExample(WorkspaceServiceGeneric):
    """
    This is a demonstration workspace service that can read and write HF json files from a local directory, and make them available for use
    via the integration.
    """

    def __init__(self, config: dict) -> None:
        """
        config.base_path: The directory where the HF json files are stored.
        """
        super().__init__(config)

    def ListWorkspaces(self, request: workspace_pb2.ListWorkspacesRequest, context) -> workspace_pb2.ListWorkspacesResponse:
        """List json files inside the base directory"""

        workspaces = []
        for filename in os.listdir(self.config["base_path"]):
            if filename.endswith('.json'):
                id = filename[:-5]
                workspaces.append(workspace_pb2.Workspace(id=id, name=id))

        return workspace_pb2.ListWorkspacesResponse(workspaces=workspaces)

    def GetWorkspace(self, request: workspace_pb2.GetWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """
        Get workspace
        """

        if os.path.exists(os.path.join(self.config["base_path"], request.workspace_id + '.json')):
            return workspace_pb2.Workspace(id=request.workspace_id, name=request.workspace_id)
        else:
            raise RuntimeError("no such workspace")

    def CreateWorkspace(self, request: workspace_pb2.CreateWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """
        Create a new workspace
        """

        # Create a new workspace
        with open(os.path.join(self.config["base_path"], request.workspace.name + '.json'), 'w') as f:
            f.write("{}")

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

        with open(os.path.join(self.config["base_path"], request.workspace_id + '.json'), 'w', encoding="utf8") as f:
            json.dump(hf_json, f, indent=2)

        return workspace_pb2.ImportWorkspaceResponse()

    def ExportWorkspace(self, request: workspace_pb2.ExportWorkspaceRequest, context) -> workspace_pb2.ExportWorkspaceResponse:
        """
        Exports a workspace from the integration, importing it into Studio
        """
        # Read the HF json data from the file
        with open(os.path.join(self.config["base_path"], request.workspace_id + '.json'), 'r') as f:
            return workspace_pb2.ExportWorkspaceResponse(
                data_format=config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON,
                format_options=self.format_options,
                data=gzip.compress(json.dumps(f.read()).encode('utf-8')),
            )
