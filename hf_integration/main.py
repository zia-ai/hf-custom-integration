"""
Main script

Config provided as command line args, the keys and the values are separated by :: 
and the key-value pairs are sepeated by ,
"""
import sys, json, threading, gc
from typing import Any, Dict, Optional
from hf_integration.humanfirst.protobuf.external_integration.v1alpha1 import workspace_pb2
# from .models.huggingface.intent_entity import IntentEntityPipeline, Trainer
from dataclasses import dataclass

from .humanfirst.protobuf.external_integration.v1alpha1 import discovery_pb2, discovery_pb2_grpc, models_pb2, models_pb2_grpc, workspace_pb2, workspace_pb2_grpc
from .humanfirst.protobuf.playbook.data.config.v1alpha1 import config_pb2

# from .model_service import ModelService
from hf_integration.workspace_generic import WorkspaceServiceGeneric
from hf_integration.workspace_service import WorkspaceService
from hf_integration.workspace_clu import WorkspaceServiceCLU
from hf_integration.workspace_example import WorkspaceServiceExample

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
    import grpc, logging, time
    from concurrent import futures
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    discovery_pb2_grpc.add_DiscoveryServicer_to_server(DiscoveryService(), grpc_server)
    # models_pb2_grpc.add_ModelsServicer_to_server(ModelService(), grpc_server) # Work In Progress

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
    if integration == "generic":
        intg = WorkspaceServiceGeneric(config=config)
    elif integration == "clu":
        intg = WorkspaceServiceCLU(config=config)
    elif integration == "example":
        intg = WorkspaceServiceExample(config=config)
    else:
        raise RuntimeError("Unrecognosed integration")

    workspace_pb2_grpc.add_WorkspacesServicer_to_server(WorkspaceService(integration=intg), grpc_server)

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

    logging.error("grpc server ready on %s" % addr)

    try:
        while True:
            time.sleep(24 * 3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    main(sys.argv[1:])
