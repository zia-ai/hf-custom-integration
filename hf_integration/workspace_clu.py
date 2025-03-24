"""
Handles workspaces during import and export
"""

# standard import
import json
import gzip
import io
import logging.config
import os
from datetime import datetime
import sys

# 3rd party imports
from pythonjsonlogger import jsonlogger

# custom imports
from hf_integration.workspace_generic import WorkspaceServiceGeneric
from hf_integration.clu_apis import clu_apis
from hf_integration.clu_converters import clu_converter
from .humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

API_VERSION = "2023-04-01"

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

DEFAULT_CONFIDENCE_THRESHOLD = 0.0

DEFAULT_NORMALIZE_CASING = False

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

# Add JSON formatter to the handlers
def add_json_formatter_to_handlers():
    json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(json_formatter)

# Apply JSON formatter
add_json_formatter_to_handlers()

# create logger
logger = logging.getLogger('custom_integration.workspace_clu')

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
        self.workspace_path = os.path.join(self.config["project_path"],"workspaces")

        # Check for language code support
        if self.config["clu_language"] in CLU_SUPPORTED_LANGUAGE_CODES:
            self.language = self.config["clu_language"]
        else:
            raise RuntimeError(f'{self.config["clu_language"]} is not supported by CLU')
        
        self.multilingual = {"True": True, "False": False}[self.config["clu_multilingual"]]

        # check for delimiter
        if "workspace_delimiter" in self.config:
            if self.config["workspace_delimiter"] != "":
                self.format_options.hierarchical_delimiter=self.config["workspace_delimiter"]
        
        # check for confidence threshold
        self.confidence_threshold = float(DEFAULT_CONFIDENCE_THRESHOLD)
        if "clu_confidence_threshold" in self.config:
            if self.config["clu_confidence_threshold"] != "":
                # has to be a float value
                self.confidence_threshold = float(self.config["clu_confidence_threshold"])
        
        # check for normalize casing
        self.normalize_casing = DEFAULT_NORMALIZE_CASING
        if "clu_normalize_casing" in self.config:
            if self.config["clu_normalize_casing"] != "":
                self.normalize_casing = {"True": True, "False": False}[self.config["clu_normalize_casing"]]


    def _write_json(self,path: str, data: dict ) -> None:
        with open(path,mode="w",encoding="utf8") as f:
            json.dump(data,f,indent=2)

    def ListWorkspaces(self, request: workspace_pb2.ListWorkspacesRequest, context) -> workspace_pb2.ListWorkspacesResponse:
        """List Workspaces"""

        logger.info("ListWorkspaces")
        # logger.info(request)

        workspaces = []
        for project in self.clu_api.list_projects():
                workspaces.append(workspace_pb2.Workspace(id=project, name=project))

        return workspace_pb2.ListWorkspacesResponse(workspaces=workspaces)

    def GetWorkspace(self, request: workspace_pb2.GetWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """Get workspace"""

        logger.info("GetWorkspace")

        # logger.info(request)

        if request.workspace_id in self.clu_api.list_projects():
            return workspace_pb2.Workspace(id=request.workspace_id, name=request.workspace_id)
        else:
            raise RuntimeError("no such workspace")

    def CreateWorkspace(self, request: workspace_pb2.CreateWorkspaceRequest, context) -> workspace_pb2.Workspace:
        """
        Create a new workspace
        """
        logger.info("CreateWorkspace")
        # logger.info(request)
        self.clu_api.clu_create_project(project_name=request.workspace.name,
                                        des = request.workspace.description,
                                        language=self.language,
                                        confidence_threshold=self.confidence_threshold,
                                        normalize_casing=self.normalize_casing,
                                        multilingual=self.multilingual,
                                        )

        return workspace_pb2.Workspace(id=request.workspace.name, name=request.workspace.name)

    def GetImportParameters(self, request: workspace_pb2.GetImportParametersRequest, context) -> workspace_pb2.GetImportParametersResponse:
        """
        Indicate the data format in which the workspace data should be provided

        In this case, we specifically request the HF json format
        """
        logger.info("GetImportParameters")
        # logger.info(request)
        # logger.info(request.language_code)
        return workspace_pb2.GetImportParametersResponse(data_format=self.data_format, format_options=self.format_options)

    def ImportWorkspace(self, request: workspace_pb2.ImportWorkspaceRequest, context) -> workspace_pb2.ImportWorkspaceResponse:
        """
        Import a workspace into the integration, from the provided data exported from Studio
        """
        logger.info("ImportWorkspace")
        # logger.info(request)
        logger.info(f"Hierarchical Delimiter: {request.format_options.hierarchical_delimiter}")
        # logger.info(request.language_code)
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        project_name = self.clu_api._remove_non_alphanumeric(input_string=request.workspace_id)

        if isinstance(context,dict):
            if "hf_file_path" in context and "clu_file_path" in context:
                hf_file_path = context["hf_file_path"]
                clu_file_path = context["clu_file_path"]
            else:
                raise RuntimeError("hf_file_path and clu_file_path not present in the context")
        else:
            hf_file_path = os.path.join(self.workspace_path,"import",f"{timestamp}_hf_{request.namespace}_{project_name}.json")
            clu_file_path = os.path.join(self.workspace_path,"import",f"{timestamp}_clu_{request.namespace}_{project_name}.json")

        # Write the HF json data to a file (in the real world, you would want to make sure there are no path injection attempts)
        # Decompress the gzip data
        with gzip.open(io.BytesIO(request.data), 'rb') as f:
            uncompressed_data = f.read()
        
        hf_json = json.loads(uncompressed_data.decode('utf-8'))

        self._write_json(
            path = hf_file_path,
            data = hf_json)

        project_metadata = self.clu_api.get_project_metadata(project_name=project_name)
        clu_json = {
            "projectFileVersion": API_VERSION,
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

        clu_json = self.clu_converter.hf_to_clu_process(
            hf_json=hf_json,
            clu_json=clu_json,
            delimiter=self.format_options.hierarchical_delimiter,
            language=self.language)

        self._write_json(
            path = clu_file_path,
            data = clu_json)

        self.clu_api.import_project(clu_json=clu_json,
                                    project_name=project_name)

        return workspace_pb2.ImportWorkspaceResponse()

    def ExportWorkspace(self, request: workspace_pb2.ExportWorkspaceRequest, context) -> workspace_pb2.ExportWorkspaceResponse:
        """
        Exports a workspace from the integration, importing it into Studio
        """

        logger.info("ExportWorkspace")
        # logger.info(request)
        # logger.info(request.language_code)

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        clu_project = self.clu_api.export_project(project_name=request.workspace_id)
        self._write_json(
            path=os.path.join(self.workspace_path,"export",f"{timestamp}_clu_{request.namespace}_{request.workspace_id}.json"),
            data = clu_project)

        hf_json = self.clu_converter.clu_to_hf_process(
            clu_json=clu_project,
            delimiter=self.format_options.hierarchical_delimiter,
            language=self.language)

        self._write_json(
            path=os.path.join(self.workspace_path,"export",f"{timestamp}_hf_{request.namespace}_{request.workspace_id}.json"),
            data = hf_json)

        # Read the HF json data from the file
        return workspace_pb2.ExportWorkspaceResponse(
            data_format=config_pb2.IntentsDataFormat.INTENTS_FORMAT_HF_JSON,
            format_options=self.format_options,
            data=gzip.compress(json.dumps(hf_json).encode('utf-8')),
        )
