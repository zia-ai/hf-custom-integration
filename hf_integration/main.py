"""
Main script

Config provided as command line args, the keys and the values are separated by :: 
and the key-value pairs are separated by ,
"""

# standard imports
import json
import sys
import logging
import os
from datetime import datetime
import grpc
import time

# custom imports
from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2_grpc, workspace_pb2_grpc
from hf_integration.model_service import ModelService
from hf_integration.model_generic import ModelServiceGeneric
from hf_integration.model_clu import ModelServiceCLU

from hf_integration.workspace_generic import WorkspaceServiceGeneric
from hf_integration.workspace_service import WorkspaceService
from hf_integration.workspace_clu import WorkspaceServiceCLU
from hf_integration.workspace_example import WorkspaceServiceExample

MAX_MESSAGE_LENGTH = 8000000

def setup_logging(log_file_path):
    # Remove all existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            # Remove StreamHandler to prevent double logging
            # logging.StreamHandler()  # You can comment this out if you don't want logs in the terminal
        ]
    )

    # Configure grpc logging
    grpc_logger = logging.getLogger('grpc')
    grpc_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    grpc_logger.addHandler(file_handler)

    return log_file_path

def redirect_output_to_log(log_file_path):
    # Redirect stdout and stderr to the log file
    log_file = open(log_file_path, 'a')
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

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

    # Configure the root logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(config["project_path"],"hf_integration","logs",f"{timestamp}.log")
    setup_logging(log_file_path)

    # Redirect stdout and stderr
    redirect_output_to_log(log_file_path)

    # Create a logger object
    logger = logging.getLogger(__name__)

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
