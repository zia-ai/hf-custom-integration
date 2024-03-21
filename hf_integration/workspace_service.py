"""
Handles workspaces during import and export
"""

# custom imports
from hf_integration.workspace_generic import WorkspaceServiceGeneric
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2, workspace_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

class WorkspaceService():
    """
    This is a workspace service that can read and write HF json files from a custom integration, and make them available for use
    via the integration.
    """

    def __init__(self, integration: WorkspaceServiceGeneric) -> None:
        self.integration = integration
    
    def ListWorkspaces(self, request: workspace_pb2.ListWorkspacesRequest, context) -> workspace_pb2.ListWorkspacesResponse:
        """List workspaces present in the integration"""

        return self.integration.ListWorkspaces(request=request, context=context)

    def GetWorkspace(self, request: workspace_pb2.GetWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """Get a workspace"""
        
        return self.integration.GetWorkspace(request=request, context=context)

    def CreateWorkspace(self, request: workspace_pb2.CreateWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """
        Create a new workspace
        """

        return self.integration.CreateWorkspace(request=request, context=context)

    def GetImportParameters(self, request: workspace_pb2.GetImportParametersRequest, context) -> workspace_pb2.GetImportParametersResponse:
        """
        Get Import parameters
        """

        return self.integration.GetImportParameters(request=request, context=context)

    def ImportWorkspace(self, request: workspace_pb2.ImportWorkspaceRequest, context) -> workspace_pb2.ImportWorkspaceResponse:
        """
        Import a workspace into the integration, from the provided data exported from Studio
        """

        return self.integration.ImportWorkspace(request=request, context=context)

    def ExportWorkspace(self, request: workspace_pb2.ExportWorkspaceRequest, context) -> workspace_pb2.ExportWorkspaceResponse:
        """
        Exports a workspace from the integration, importing it into Studio
        """
        
        return self.integration.ExportWorkspace(request=request, context=context)
