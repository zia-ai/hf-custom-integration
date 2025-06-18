"""
Main script

Config provided as command line args, the keys and the values are separated by :: 
and the key-value pairs are separated by ,
"""

# standard imports
import json
import sys
import logging
import logging.config
import os
from datetime import datetime
import grpc
import time
from pythonjsonlogger import jsonlogger
import re

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2_grpc, workspace_pb2_grpc
from hf_integration.model_service import ModelService
from hf_integration.model_generic import ModelServiceGeneric
from hf_integration.model_clu import ModelServiceCLU

from hf_integration.workspace_generic import WorkspaceServiceGeneric
from hf_integration.workspace_service import WorkspaceService
from hf_integration.workspace_clu import WorkspaceServiceCLU
from hf_integration.workspace_example import WorkspaceServiceExample

MAX_MESSAGE_LENGTH = 200 << 20

def cleanup_logging_handlers():
    # Remove all existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()


def redirect_output_to_log(log_file_path):
    # Redirect stdout and stderr to the log file
    log_file = open(log_file_path, 'a')
    # os.dup2(log_file.fileno(), sys.stdout.fileno()) # This statement would push even the print statements into log file
    # Any gRPC or SSL errors redirected to log file
    os.dup2(log_file.fileno(), sys.stderr.fileno())

# Clean Up existing logging handlers
cleanup_logging_handlers()

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
logger = logging.getLogger('custom_integration.main')

if log_file_enable == "TRUE":
    # this is necessary because SSL logs are printed onto the console.
    # Python logging configuration doesn't seem to overrise SSL logging configuration.
    # Hence this helps in redirecting the console output to log files
    redirect_output_to_log(log_file_path=path_to_save_log)

class DiscoveryService(discovery_pb2_grpc.DiscoveryServicer):
    def __init__(self) -> None:
        super().__init__()

    # Discovery
    def GetCapabilities(self, request: discovery_pb2.GetCapabilitiesRequest, context) -> discovery_pb2.GetCapabilitiesResponse:
        # Indicate that this service supports training and running models
        return discovery_pb2.GetCapabilitiesResponse(
            supports_models=True,
            supports_workspaces=True,
        )

def main(args):
    from concurrent import futures

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=100), options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])

    discovery_pb2_grpc.add_DiscoveryServicer_to_server(DiscoveryService(), grpc_server)

    integration = args[2]
    config = args[3]

    # Splitting the string by commas
    pairs = config.split(',')

    # Creating a dictionary from key-value pairs
    result_dict = {}
    for pair in pairs:
        key, value = pair.split('::')
        result_dict[key] = value

    config = result_dict

    # set project path
    config["project_path"] = here

    if integration == "generic":
        workspace_intg = WorkspaceServiceGeneric(config=config)
        model_intg = ModelServiceGeneric(config=config)
    elif integration == "clu":
        workspace_intg = WorkspaceServiceCLU(config=config)
        model_intg = ModelServiceCLU(config=config)
    # elif integration == "example":
    #     workspace_intg = WorkspaceServiceExample(config=config)
    else:
        raise RuntimeError("Unrecognized integration")

    workspace_pb2_grpc.add_WorkspacesServicer_to_server(WorkspaceService(integration=workspace_intg), grpc_server)
    models_pb2_grpc.add_ModelsServicer_to_server(ModelService(integration=model_intg), grpc_server)

    # Load the mtls keypair
    keypair_path = args[0]
    with open(keypair_path, 'r') as f:
        keypair = json.load(f)
        server_certificate = bytes(keypair['local_keypair']['certificate'], encoding='utf8')
        server_key = bytes(keypair['local_keypair']['private_key'], encoding='utf8')
        client_cert = bytes(keypair['remote_certificate'], encoding='utf8')

    credentials = grpc.ssl_server_credentials(
        [(server_key, server_certificate)],
        root_certificates=client_cert,
        require_client_auth=True)

    addr = args[1]
    grpc_server.add_secure_port(addr, credentials)
    grpc_server.start()

    logger.info("grpc server ready on %s", addr)

    try:
        while True:
            time.sleep(24 * 3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)

if __name__ == '__main__':
    main(sys.argv[1:])
